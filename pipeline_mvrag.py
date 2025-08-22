import PIL
import einops
import inspect
from torchvision.transforms import  v2
from typing import List, Optional
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, CLIPImageProcessor
from diffusers import AutoencoderKL, DiffusionPipeline
from diffusers.utils import (
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
    logging,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.configuration_utils import FrozenDict
from diffusers.schedulers import DDIMScheduler

from modules.mv_unet import MultiViewUNetModel
from utils import *
from modules.f_score import SimilarityModel
from modules.resampler import Resampler

logger = logging.get_logger(__name__)


class MVRAGPipeline(DiffusionPipeline):
    def __init__(
            self,
            vae: AutoencoderKL,
            unet: MultiViewUNetModel,
            tokenizer: CLIPTokenizer,
            text_encoder: CLIPTextModel,
            scheduler: DDIMScheduler,
            feature_extractor: CLIPImageProcessor,
            image_encoder: CLIPVisionModel,
            resampler: Resampler,
            requires_safety_checker: bool = False,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:  # type: ignore
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "  # type: ignore
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate(
                "steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:  # type: ignore
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate(
                "clip_sample not set", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)
        self.similarity_model = SimilarityModel()
        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            resampler=resampler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

        self.transform = v2.Compose([
            pad_to_square,
            v2.Resize(size=256),
        ])

    def to(self, device, **kwargs):
        super().to(device, **kwargs)
        if self.similarity_model is not None:
            self.similarity_model.to(device)
        return self


    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding.

        When this option is enabled, the VAE will split the input tensor into tiles to compute decoding and encoding in
        several steps. This is useful to save a large amount of memory and to allow the processing of larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        Note that offloading happens on a submodule basis. Memory savings are higher than with
        `enable_model_cpu_offload`, but performance is lower.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.14.0"):
            from accelerate import cpu_offload
        else:
            raise ImportError(
                "`enable_sequential_cpu_offload` requires `accelerate v0.14.0` or higher"
            )

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            cpu_offload(cpu_offloaded_model, device)

    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError(
                "`enable_model_offload` requires `accelerate v0.17.0` or higher."
            )

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        hook = None
        for cpu_offloaded_model in [self.text_encoder, self.unet, self.vae]:
            _, hook = cpu_offload_with_hook(
                cpu_offloaded_model, device, prev_module_hook=hook
            )

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                    hasattr(module, "_hf_hook")
                    and hasattr(module._hf_hook, "execution_device")
                    and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(
            self,
            prompt,
            device,
            do_classifier_free_guidance: bool,
            negative_prompt=None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
            if prompt.endswith('.'):
                prompt = prompt[:-1]
            prompt = [prompt + ", 3d asset"]

        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
            prompt = [p[:-1] + ", 3d asset" if p.endswith(".") else p + ", 3d asset" for p in prompt]
        else:
            raise ValueError(
                f"`prompt` should be either a string or a list of strings, but got {type(prompt)}."
            )

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1: -1]
            )
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
        ):
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        prompt_embeds = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt] * batch_size
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if (
                    hasattr(self.text_encoder.config, "use_attention_mask")
                    and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )[0]
            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=self.text_encoder.dtype, device=device
            )
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(
            self,
            batch_size,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
            latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def encode_images(self, images, device):
        dtype = next(self.image_encoder.parameters()).dtype

        ret_images = [self.transform(ret) for ret in images]
        images_proc = self.feature_extractor(ret_images, return_tensors="pt").pixel_values
        images_proc = images_proc.to(device=device, dtype=dtype)
        clip_images = self.image_encoder(images_proc, output_hidden_states=True).hidden_states[-2]

        neg_images = torch.zeros_like(images_proc, device=device)
        clip_images_neg = self.image_encoder(neg_images, output_hidden_states=True).hidden_states[-2]

        image_embeds = torch.cat([clip_images_neg, clip_images], dim=0)
        image_tokens = self.resampler(image_embeds)
        return image_tokens


    @torch.no_grad()
    def __call__(
            self,
            prompt: str = "",
            images = None,
            height: int = 256,
            width: int = 256,
            elevation: int = 0,
            azimuth_start:int = 0,
            num_inference_steps: int = 50,
            num_initial_steps: int = 10,
            guidance_scale: float = 7.0,
            negative_prompt: str = "",
            eta: float = 0.0,
            output_type: Optional[str] = "numpy",  # pil, numpy
            num_frames: int = 4,
            device=torch.device("cuda:0"),
            seed: Optional[int] = None,
    ):
        self.unet = self.unet.to(device=device)
        self.vae = self.vae.to(device=device)
        self.text_encoder = self.text_encoder.to(device=device)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        bs = len(prompt) if isinstance(prompt, list) else 1
        bs = bs * 2 if do_classifier_free_guidance else bs
        assert isinstance(images, list) and len(images) > 0 and isinstance(images[0], PIL.Image.Image)
        self.image_encoder = self.image_encoder.to(device=device)
        ret_eval_embs = self.similarity_model.get_embeddings(images)
        image_tokens = self.encode_images(images, device)
        image_tokens = einops.rearrange(image_tokens, "(b n) c f -> b (n c) f", b=bs)
        image_tokens = torch.repeat_interleave(image_tokens, num_frames, dim=0)

        prompt_embeds = self._encode_prompt(
            prompt=prompt,
            device=device,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)
        # Prepare latent variables
        latents: torch.Tensor = self.prepare_latents(
            num_frames,
            4,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator=generator,
        )
        # Get camera
        camera = get_camera(num_frames, elevation=elevation, azimuth_start=azimuth_start).to(dtype=latents.dtype, device=device)

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # initial forward pass for fusion coefficient
        base_out = self._sample(camera=camera,
                               device=device,
                               extra_step_kwargs=extra_step_kwargs,
                               guidance_scale=guidance_scale,
                               latents=latents.clone(),
                               num_frames=num_frames,
                               num_inference_steps=num_initial_steps,
                               prompt_embeds=prompt_embeds,
                               )
        embs = self.similarity_model.get_embeddings(base_out)
        scale = max(0, self.similarity_model.get_similarity_score(embs, ret_eval_embs))

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            latents = self._sample(camera=camera,
                                   device=device,
                                   extra_step_kwargs=extra_step_kwargs,
                                   guidance_scale=guidance_scale,
                                   image_tokens=image_tokens,
                                   latents=latents,
                                   num_frames=num_frames,
                                   num_inference_steps=num_inference_steps,
                                   progress_bar=progress_bar,
                                   prompt_embeds=prompt_embeds,
                                   scale=scale,
                                   )
        # Post-processing
        images = latents.cpu().permute(0, 2, 3, 1).float().numpy()
        if output_type == "pil":
            images = self.numpy_to_pil(images)
        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        return images

    def _sample(self, device,
                extra_step_kwargs, guidance_scale,
                latents,
                num_inference_steps, prompt_embeds,
                progress_bar=None,
                num_frames=4,
                camera=None,
                do_classifier_free_guidance=True,
                image_tokens=None,
                scale=1.0
                ):
        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            multiplier = 2 if do_classifier_free_guidance else 1
            latent_model_input = torch.cat([latents] * multiplier)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            unet_inputs = {
                'x': latent_model_input,
                'timesteps': torch.tensor([t] * num_frames * multiplier, dtype=latent_model_input.dtype, device=device),
                'context': torch.repeat_interleave(prompt_embeds, num_frames, dim=0),
                'num_frames': num_frames,
                'camera': torch.cat([camera] * multiplier),
                'images_tokens': image_tokens,
                'scale': scale
            }
            # predict the noise residual
            noise_pred = self.unet.forward(**unet_inputs)

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents: torch.Tensor = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs, return_dict=False
            )[0]

            if (i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            )) and progress_bar is not None:
                progress_bar.update()
        latents = self.decode_latents(latents)
        return latents
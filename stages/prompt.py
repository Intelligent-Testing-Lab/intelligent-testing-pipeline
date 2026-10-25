from pathlib import Path

import openai
from omegaconf import DictConfig

from stage import Stage


class PromptStage(Stage):
    """Caption the RGB input with a local vision-LLM and write ``prompt.txt``.

    On any failure (endpoint down, model error) it falls back to the generic
    prompt so the downstream pipeline can still proceed.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config, stage="prompt")

    def _run(self) -> None:
        cfg = self.config.stage
        out = Path(self.config.run.prompt_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        video_input = Path(self.config.run.rgb_input)

        client = openai.OpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            timeout=cfg.timeout,
        )
        try:
            prompt = self._generate_prompt(client, video_input, cfg)
        except Exception as e:  # noqa: BLE001 - fall back to the generic prompt
            self.logger.warning("Prompt generation failed (%s); using generic prompt", e)
            prompt = cfg.generic_prompt

        out.write_text(prompt)
        self.logger.info("Wrote prompt (%d chars) to %s", len(prompt), out)

    @staticmethod
    def _generate_prompt(client: "openai.OpenAI", video_input: Path, cfg: DictConfig) -> str:
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=[
                {"role": "system", "content": cfg.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": cfg.user_text},
                        {"type": "video_url", "video_url": {"url": f"file://{video_input}"}},
                    ],
                },
            ],
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
        )
        return resp.choices[0].message.content

    def verify_completion(self) -> bool:
        out = Path(self.config.run.prompt_output)
        return out.exists() and out.stat().st_size > 0

"""
Text-to-Speech через Edge TTS.

Мультиязычный: русский, английский, китайский.
Качает голос через интернет (edge-tts).
"""
import asyncio
import logging
import tempfile
from pathlib import Path

import edge_tts

from .config import Config

logger = logging.getLogger(__name__)


class TextToSpeech:
    """Озвучка текста через Edge TTS."""

    # Популярные голоса
    VOICES = {
        # Русские
        "ru_male": "ru-RU-DmitryNeural",
        "ru_female": "ru-RU-SvetlanaNeural",
        # Английские
        "en_male": "en-US-GuyNeural",
        "en_female": "en-US-JennyNeural",
        # Китайские
        "zh_male": "zh-CN-YunxiNeural",
        "zh_female": "zh-CN-XiaoxiaoNeural",
    }

    def __init__(self, voice: str = None, rate: str = "+0%"):
        """
        voice: имя голоса (например, ru-RU-DmitryNeural)
        rate: скорость (+0% по умолчанию, +20% быстрее)
        """
        self.voice = voice or Config.TTS_VOICE
        self.rate = rate
        logger.info("TTS: голос=%s, скорость=%s",
                    self.voice, self.rate)

    async def _synthesize_async(
        self, text: str, output_path: str,
    ) -> None:
        """Асинхронный вызов edge-tts."""
        communicate = edge_tts.Communicate(
            text, self.voice, rate=self.rate)
        await communicate.save(output_path)

    def synthesize_to_file(
        self, text: str, output_path: str = None,
    ) -> str:
        """
        Синтез текста в MP3-файл.

        text: текст для озвучки
        output_path: путь (если None — временный)

        Возвращает путь к файлу.
        """
        if not text or not text.strip():
            logger.warning("Пустой текст — нечего озвучивать")
            return ""

        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(
                suffix=".mp3", delete=False)
            output_path = tmp.name
            tmp.close()

        try:
            asyncio.run(
                self._synthesize_async(text, output_path))
            logger.info("Озвучено: %s (%d байт)",
                        output_path,
                        Path(output_path).stat().st_size)
            return output_path
        except Exception as e:
            logger.error("Ошибка TTS: %s", e)
            return ""

    def speak(self, text: str) -> bool:
        """
        Синтез + воспроизведение через sounddevice/pydub.

        Возвращает True при успехе.
        """
        if not text or not text.strip():
            return False

        mp3_path = self.synthesize_to_file(text)
        if not mp3_path:
            return False

        try:
            # Конвертируем MP3 → WAV через pydub + ffmpeg
            from pydub import AudioSegment
            import sounddevice as sd
            import numpy as np

            audio = AudioSegment.from_mp3(mp3_path)
            samples = np.array(
                audio.get_array_of_samples(),
                dtype=np.int16,
            )

            # Стерео → моно (если нужно)
            if audio.channels == 2:
                samples = samples.reshape(-1, 2).mean(axis=1)
                samples = samples.astype(np.int16)

            sd.play(samples, samplerate=audio.frame_rate)
            sd.wait()

            # Удаляем временный файл
            try:
                Path(mp3_path).unlink()
            except Exception:
                pass

            return True

        except Exception as e:
            logger.error("Ошибка воспроизведения: %s", e)
            return False

    def set_voice(self, voice_name: str) -> None:
        """Сменить голос."""
        self.voice = voice_name
        logger.info("TTS голос изменён: %s", voice_name)
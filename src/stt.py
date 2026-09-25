"""
Speech-to-Text через faster-whisper.

Распознаёт русский, английский, китайский.
Работает локально (модель скачивается один раз).
"""
import logging
import tempfile
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

from .config import Config

logger = logging.getLogger(__name__)


class SpeechToText:
    """Распознавание речи через faster-whisper."""

    def __init__(
        self,
        model_size: str = None,
        language: str = None,
        device: str = None,
    ):
        """
        model_size: tiny / base / small / medium / large-v3
        language: ru / en / zh / None (автоопределение)
        device: cuda / cpu
        """
        self.model_size = model_size or Config.WHISPER_MODEL
        self.language = language or Config.WHISPER_LANGUAGE
        self.device = device or Config.WHISPER_DEVICE

        logger.info(
            "Загрузка Whisper: %s (%s, %s)",
            self.model_size, self.language, self.device)

        try:
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type="float16"
                if self.device == "cuda" else "int8",
            )
            logger.info("Whisper загружен")
        except Exception as e:
            logger.warning(
                "Не удалось загрузить Whisper на %s: %s. "
                "Падаю на CPU.",
                self.device, e)
            self.device = "cpu"
            self.model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
            )

    def transcribe_file(self, audio_path: str) -> str:
        """
        Распознать аудиофайл.

        audio_path: путь к WAV/MP3/OGG.
        Возвращает текст (пустая строка, если ничего не распознано).
        """
        segments, info = self.model.transcribe(
            audio_path,
            language=self.language if self.language else None,
            beam_size=5,
            vad_filter=True,
        )

        text_parts = [seg.text.strip() for seg in segments]
        text = " ".join(text_parts).strip()

        logger.info(
            "Распознано (%s, %.2f): %s",
            info.language, info.language_probability, text[:80])
        return text

    def transcribe_array(
        self, audio: np.ndarray, sample_rate: int = 16000,
    ) -> str:
        """
        Распознать numpy-массив с аудио (int16 или float32).

        sample_rate: частота (16000 по умолчанию).
        Возвращает текст.
        """
        # faster-whisper принимает float32 [-1, 1]
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        segments, info = self.model.transcribe(
            audio,
            language=self.language if self.language else None,
            beam_size=5,
            vad_filter=True,
        )

        text_parts = [seg.text.strip() for seg in segments]
        return " ".join(text_parts).strip()

    def record_and_transcribe(
        self,
        audio_stream,
        duration_sec: float = 8.0,
        sample_rate: int = 16000,
        silence_threshold: float = 0.01,
        silence_duration: float = 1.5,
    ) -> str:
        """
        Записать с микрофона и распознать.

        audio_stream: sounddevice.InputStream
        duration_sec: максимальная длительность записи
        silence_threshold: уровень тишины (RMS)
        silence_duration: сколько секунд тишины = конец фразы

        Возвращает распознанный текст.
        """
        import time

        frames = []
        chunk_size = 1280  # 80 мс при 16 кГц
        total_samples = int(duration_sec * sample_rate)
        silence_samples = 0
        max_silence_samples = int(silence_duration * sample_rate)
        recorded = 0

        logger.info("Говорите... (макс %.1f сек)", duration_sec)

        while recorded < total_samples:
            frame, overflowed = audio_stream.read(chunk_size)
            if frame.ndim > 1:
                frame = frame[:, 0]

            frames.append(frame.copy())
            recorded += len(frame)

            # RMS
            rms = np.sqrt(
                np.mean(frame.astype(np.float32) ** 2)) / 32768.0

            if rms < silence_threshold:
                silence_samples += len(frame)
                if silence_samples >= max_silence_samples:
                    logger.info(
                        "Тишина %.1f сек — конец фразы",
                        silence_duration)
                    break
            else:
                silence_samples = 0

        if not frames:
            logger.warning("Ничего не записано")
            return ""

        audio = np.concatenate(frames)

        # Сохраняем во временный файл (для отладки)
        try:
            import scipy.io.wavfile as wav
            tmp = tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False)
            wav.write(tmp.name, sample_rate, audio)
            logger.debug("Аудио сохранено: %s", tmp.name)
        except Exception as e:
            logger.debug("Не удалось сохранить WAV: %s", e)

        text = self.transcribe_array(audio, sample_rate)
        logger.info("Распознано: %s", text[:100])
        return text
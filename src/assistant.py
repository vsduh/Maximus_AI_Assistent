"""
Maximus AI Assistant — главный цикл.

Wake word → STT → LLM → TTS.
"""
import logging
import sys
import time

import numpy as np
import sounddevice as sd

from .config import Config
from .llm import DeepSeekLLM
from .stt import SpeechToText
from .tts import TextToSpeech
from .wake_word import WakeWordDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("assistant")


class MaximusAssistant:
    """Голосовой ассистент Максимус."""

    def __init__(self):
        # 1. Проверка конфига
        errors = Config.validate()
        if errors:
            for e in errors:
                logger.error("Config: %s", e)
            sys.exit(1)

        # 2. Проверка модели wake word
        ww_path = Config.wake_word_path()
        if not ww_path.exists():
            logger.error(
                "Модель wake word не найдена: %s",
                ww_path)
            logger.error(
                "Сначала обучи: python scripts/train_wake_word.py")
            sys.exit(1)

        # 3. Инициализация компонентов
        logger.info("=" * 60)
        logger.info("Инициализация Максимуса...")
        logger.info("=" * 60)

        self.wake_word = WakeWordDetector()
        self.stt = SpeechToText()
        self.tts = TextToSpeech()
        self.llm = DeepSeekLLM()

        # 4. Микрофон
        self.stream = sd.InputStream(
            samplerate=Config.MIC_SAMPLE_RATE,
            channels=Config.MIC_CHANNELS,
            dtype="int16",
            blocksize=1280,  # 80 мс при 16 кГц
        )

        self.running = True
        logger.info("Готов к работе")
        logger.info("Скажи «Максимус», чтобы активировать")
        logger.info("Ctrl+C — выход")
        logger.info("=" * 60)

    def run(self):
        """Главный цикл."""
        try:
            self.stream.start()

            while self.running:
                # 1. Ждём wake word
                if not self.wake_word.wait_for_wake_word(
                        self.stream):
                    break

                # 2. Короткая пауза (чтобы не поймать хвост wake word)
                time.sleep(0.3)

                # 3. Слушаем вопрос (до 10 сек, тишина 1.5 сек = конец)
                question = self.stt.record_and_transcribe(
                    self.stream,
                    duration_sec=10.0,
                    sample_rate=Config.MIC_SAMPLE_RATE,
                )

                if not question or len(question.strip()) < 2:
                    logger.info("Пустой вопрос — пропуск")
                    continue

                logger.info("❓ %s", question)

                # 4. Ответ LLM (стриминг)
                logger.info("🤔 Думаю...")
                full_answer = ""

                for chunk in self.llm.chat_stream(question):
                    full_answer += chunk

                full_answer = full_answer.strip()
                if not full_answer:
                    logger.warning("Пустой ответ")
                    continue

                logger.info("💬 %s", full_answer)

                # 5. Озвучка
                logger.info("🔊 Говорю...")
                self.tts.speak(full_answer)

                # 6. Сброс wake word (следующее слово — заново)
                self.wake_word.reset()

        except KeyboardInterrupt:
            logger.info("Прервано (Ctrl+C)")
        finally:
            self.stop()

    def stop(self):
        """Остановка."""
        self.running = False
        try:
            self.stream.stop()
            self.stream.close()
        except Exception as e:
            logger.debug("stream close: %s", e)
        logger.info("Остановлено")


def main():
    try:
        assistant = MaximusAssistant()
        assistant.run()
    except Exception as e:
        logger.exception("Критическая ошибка: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
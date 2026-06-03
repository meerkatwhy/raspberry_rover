import asyncio
from enum import Enum, auto

from robot.services.listener import Listener
from robot.services.vision import Vision


class State(Enum):
    IDLE = auto()
    TRANSCRIBING = auto()
    MOVING = auto()
    DETECTING_OBJECTS = auto()
    STOPPING = auto()


class StateMachine:
    def __init__(self) -> None:
        self.listener = Listener()

        # self.vision = Vision()

        self.state = State.IDLE
        self.transcription = ""
        self.commands = []

    async def run(self) -> None:
        try:
            while True:
                match self.state:
                    case State.IDLE:
                        print("State: IDLE")
                        await self.listener.wait_wakeword()
                        self.state = State.TRANSCRIBING

                    case State.TRANSCRIBING:
                        print("State: TRANSCRIBING")

                        self.transcription = (
                            await self.listener.transcribe()
                        )

                        if self.transcription:
                            print(f"Transcription: {self.transcription}")
                            self._parse_transcript()
                            self.state = State.MOVING
                        else:
                            self.state = State.IDLE

                    case State.MOVING:
                        await asyncio.sleep(0)
                        self.state = State.IDLE

                    case State.DETECTING_OBJECTS:
                        await asyncio.sleep(0)

                    case State.STOPPING:
                        break

        finally:
            await self.listener.close()

    def _parse_transcript(self) -> None:
        # parse transcript and generate correct commands to fill self.commands
        pass

    def stop(self) -> None:
        self.state = State.STOPPING

mysm = StateMachine()
asyncio.run(mysm.run())
import asyncio
from robot.statemachine import StateMachine

if __name__ == "__main__":
    statemachine = StateMachine()
    asyncio.run(statemachine.run())
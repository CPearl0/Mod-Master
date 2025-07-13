import modmaster
import modmaster.agents
import modmaster.agents.TaskAgent


def main():
    print("=" * 50)
    print("Mod Master!")
    print("=" * 50)

    task = input("What do you want Mod Master to do? (In detail!)")

    agent = modmaster.agents.TaskAgent.TaskAgent()

    print("\nProcessing task...")
    state = agent.run(task)

    print("\n" + "=" * 50)
    print("TASK REPORT")
    print("=" * 50)

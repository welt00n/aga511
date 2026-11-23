import argparse


def ask(question, default):
    shown = f" [{default}]" if default != "" else ""
    answer = input(f"{question}{shown}: ").strip()
    return answer or default


def ask_choice(question, choices, default):
    print(question)
    for number, (name, description) in enumerate(choices, 1):
        print(f"  {number}. {name}" + (f"  {description}" if description else ""))
    names = [name for name, _ in choices]
    while True:
        answer = ask("  a number or a name", default)
        if answer.isdigit() and 1 <= int(answer) <= len(names):
            return names[int(answer) - 1]
        if answer in names:
            return answer
        print(f"  not one of: {', '.join(names)}")


def ask_value(question, default, convert):
    while True:
        answer = ask(question, default)
        if answer == default:
            return answer
        try:
            convert(answer)
            return answer
        except (ValueError, argparse.ArgumentTypeError) as error:
            print(f"  {error}")


def ask_yes(question, default=False):
    return ask(question + " (y/n)", "y" if default else "n").lower().startswith("y")

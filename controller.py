# controller.py
import subprocess

def run_program1():
    subprocess.run(["python", "traffic\\data_source.py"])

def run_program2():
    subprocess.run(["python", "traffic\\vehicleDetect.py"])

def main():
    while True:
        run_program1()
        run_program2()
        user_input = input("Press 'q' to quit or any other key to continue: ")
        if user_input.lower() == "q":
            break

if __name__ == "__main__":
    main()

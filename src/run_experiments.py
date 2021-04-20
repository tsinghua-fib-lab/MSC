import os
import time
import argparse


def run_experiments(cmd, device):
    for command in cmd:
        rty_flag = 1
        retry = 0
        cmd_dvs = " -dvs {}".format(device)
        command += cmd_dvs
        print(' ----- Executing task on {} ----- '.format(device))
        while rty_flag != 0:
            rty_flag = os.system(command)
            rty_flag >>= 8
            time.sleep(3)
            retry += 1
            if retry >= 3:
                print(' -------------- Command failed -------------- ')
                print(command)
                return 0
    return 0


def get_experiments(path):
    cmd = []
    with open(path, 'r') as file:
        for line in file:
            if len(line) > 5:
                cmd.append(line.strip())
    return cmd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiments')
    parser.add_argument('-p', '--config_path', type=str,
                        default='./exp_config/xxx', help='config path')
    parser.add_argument('-dvs', '--device', type=str,
                        default='cuda:0', help='device')
    args = parser.parse_args()

    cmd = get_experiments(args.config_path)
    run_experiments(cmd, args.device)

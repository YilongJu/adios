import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--command", type=str, default="")
parser.add_argument("--gpu_ids", type=str)
parser.add_argument("--prefix", type=str)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--create', action='store_true')
# parser.add_argument('--no-cuda', dest='cuda', action='store_false')
parser.set_defaults(cuda=False)
parser.add_argument('--stop', action='store_true')
parser.add_argument('--setup', action='store_true')
# parser.add_argument('--no-stop', dest='stop', action='store_false')
parser.set_defaults(stop=False)
# parse args
args = parser.parse_args()

command = args.command
command_setup = "conda activate jpy3.8; cd ~/Github/adios"

for gpu_id in args.gpu_ids.split(","):
    cuda_str = f"CUDA_VISIBLE_DEVICES={gpu_id} " if args.cuda else ""
    command_str = "^C^C" if args.stop else command
    command_str = command_setup if args.setup else command_str

    flag = "-dmS" if args.create else "-S"
    full_command = f"" if args.create else f" -p 0 -X stuff '{cuda_str}{command_str}\\n'"
    # screen -dmS "$SESSION_NAME" "$COMMAND"
    output = f"screen {flag} {args.prefix}{gpu_id}{full_command}"
    print(output)

    # Send the output to clipboard
    """ Send the output to clipboard """

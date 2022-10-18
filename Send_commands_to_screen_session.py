import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--command", type=str, default="")
parser.add_argument("--gpu_ids", type=str)
parser.add_argument("--prefix", type=str)
parser.add_argument('--cuda', action='store_true')
# parser.add_argument('--no-cuda', dest='cuda', action='store_false')
parser.set_defaults(cuda=False)
parser.add_argument('--stop', action='store_true')
# parser.add_argument('--no-stop', dest='stop', action='store_false')
parser.set_defaults(stop=False)
# parse args
args = parser.parse_args()

command = args.command
for gpu_id in args.gpu_ids.split(","):
    cuda_str = f"CUDA_VISIBLE_DEVICES={gpu_id} " if args.cuda else ""
    if args.stop:
        command_str = "^C^C"
    else:
        command_str = command
    output = f"screen -S {args.prefix}{gpu_id} -p 0 -X stuff '{cuda_str}{command_str}\\n'"
    print(output)
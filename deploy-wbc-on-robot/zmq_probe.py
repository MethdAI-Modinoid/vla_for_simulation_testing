"""Diagnose a camera ZMQ server: subscribe to EVERYTHING and report what arrives.

Unlike the VLA clients, this subscribes to the empty prefix ("" = all topics),
so it sees traffic regardless of topic name or wire format. Use it to tell
apart the three failure modes behind a stuck "image False":

    * 0 messages          -> server down / wrong host:port / firewall
    * messages, but the topic names differ from what the client expects
    * messages arrive as a single part (msgpack) -> different server protocol

Usage:
    python zmq_probe.py --connect tcp://192.168.123.164:5555 --seconds 5
"""

import argparse
import time
from collections import Counter

import zmq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--connect", default="tcp://192.168.123.164:5555")
    ap.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args()

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.connect(args.connect)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")  # receive every topic

    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)

    print(f"[probe] listening on {args.connect} for {args.seconds:.0f}s ...")
    total = 0
    part_counts = Counter()   # how many frames each message had
    topics = Counter()        # first-frame value (topic) for multipart messages
    deadline = time.time() + args.seconds

    while time.time() < deadline:
        if not dict(poller.poll(timeout=200)):
            continue
        parts = sock.recv_multipart()
        total += 1
        part_counts[len(parts)] += 1
        if len(parts) >= 2:
            try:
                topics[parts[0].decode("utf-8")] += 1
            except UnicodeDecodeError:
                topics["<non-utf8 first frame>"] += 1

    print(f"[probe] total messages: {total}")
    if total == 0:
        print("[probe] NOTHING received -> server not publishing or unreachable.")
        print("        Check the server is running and `nc -zv <host> <port>` succeeds.")
    else:
        print(f"[probe] message shapes (frames -> count): {dict(part_counts)}")
        if topics:
            print(f"[probe] topics seen (topic -> count): {dict(topics)}")
            print("        Compare these against the client's expected topics:")
            print("        stereo/right, hand_left, hand_right")
        else:
            print("[probe] single-part messages (no topic frame) -> this is NOT the")
            print("        multipart [topic, meta, jpg] server the exporter expects.")

    sock.close(linger=0)


if __name__ == "__main__":
    main()

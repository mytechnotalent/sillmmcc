![image](https://github.com/mytechnotalent/sillmmcc/blob/main/sillmmcc.png?raw=true)

## FREE Reverse Engineering Self-Study Course [HERE](https://github.com/mytechnotalent/Reverse-Engineering-Tutorial)

<br><br>

# Serial Interactive LLM Meshtastic Chat Client
Serial Interactive LLM Meshtastic Chat Client which chats on the Primary Channel over serial.

<br><br>

### STEP 1: `python3 -m venv venv`
### STEP 2: `source venv/bin/activate`
### STEP 3: `pip install -r requirements.txt`
### STEP 4: `./sillmmcc.py <SERIAL_PORT>`

### SOURCE
```python
#!/usr/bin/env python3

"""
Serial Interactive LLM Meshtastic Chat Client 0.1.0

Usage:
    python sillmmcc.py <SERIAL_PORT>

This script sends and receives text messages over the serial interface.
It uses PyPubSub to subscribe to text messages.
"""

import sys
import time
import asyncio
import logging
import io
from contextlib import redirect_stderr

# silence transformer logs (including the pad_token_id message)
logging.getLogger("transformers").setLevel(logging.ERROR)

from pubsub import pub
from meshtastic.serial_interface import SerialInterface
from transformers import pipeline

# use a capable conversational model
local_chat = pipeline("text2text-generation", model="facebook/blenderbot-400M-distill")

# async queue for incoming messages.
message_queue = asyncio.Queue()


def run_pipeline_call(func, *args, **kwargs):
    """
    Helper function to call a given function while temporarily redirecting stderr.

    This function redirects the standard error stream to an in-memory buffer to suppress
    unwanted logging messages (e.g., messages regarding pad_token_id settings). This ensures that
    any non-critical log messages produced by the function do not clutter the console output.

    Parameters:
        func (callable): The function to be called.
        *args: Variable length argument list to pass to the function.
        **kwargs: Arbitrary keyword arguments to pass to the function.

    Side Effects:
        - Temporarily redirects stderr to an in-memory buffer.
        - Suppresses unwanted logging output from the function call.

    Returns:
        Any: The result returned by the function call func(*args, **kwargs).
    """
    with io.StringIO() as buf, redirect_stderr(buf):
        result = func(*args, **kwargs)
    return result


def onReceive(packet=None, interface=None):
    """
    Callback function to process an incoming text message when the topic "meshtastic.receive.text" is published.

    This function accepts both 'packet' and 'interface' as optional parameters to avoid errors from extra keyword
    arguments. When the Meshtastic library publishes on the "meshtastic.receive.text" topic, it may include additional
    arguments that are not needed by this function. The function extracts the text message from the packet and enqueues
    it for asynchronous processing by the LLM responder.

    Parameters:
        packet (dict, optional): A dictionary containing data about the received packet. If provided, it may include:
            - 'decoded': A dictionary that may contain a 'text' key with the text message.
            - 'fromId': (Optional) The identifier of the sender. Defaults to "unknown" if not provided.
        interface (optional): The Meshtastic interface instance that received the packet. This parameter is accepted to
            avoid pubsub errors but is not used within this function.

    Side Effects:
        - If the packet contains a text message (under 'decoded.text'), prints the sender ID and the message to standard output.
        - Enqueues a tuple containing the sender ID and the message into the global message_queue for further processing.

    Returns:
        None
    """
    if not packet:
        return

    decoded = packet.get("decoded", {})
    if "text" in decoded:
        sender = packet.get("fromId", "unknown")
        message = decoded["text"]
        print(f"\nIncoming from {sender}: {message}")
        message_queue.put_nowait((sender, message))


async def generate_llm_response(message: str) -> str:
    """
    Asynchronous function to generate a reply using the local LLM based on an input message.

    This function uses a text-generation pipeline to produce a chat response. It constructs a prompt using the
    provided message, executes the pipeline in a non-blocking manner via an executor, and then post-processes
    the generated text to ensure it does not exceed 20 words.

    Parameters:
        message (str): The input message received from a user, serving as the prompt for the LLM.

    Side Effects:
        - Executes the text-generation pipeline in a separate thread to avoid blocking the asyncio event loop.
        - May generate suppressed log output during the model inference process.

    Returns:
        str: A generated response string limited to a maximum of 20 words.
    """
    loop = asyncio.get_running_loop()
    # use the incoming message as prompt.
    prompt = message
    results = await loop.run_in_executor(
        None,
        lambda: run_pipeline_call(
            local_chat,
            prompt,
            max_length=50,
            do_sample=True,
            top_k=50,
            truncation=True  # explicitly enable truncation
        )
    )
    generated_text = results[0]["generated_text"]

    # mimit the response to a maximum of 20 words
    words = generated_text.split()
    if len(words) > 20:
        generated_text = " ".join(words[:20])
    return generated_text


async def run_llm_responder(iface):
    """
    Asynchronous function that continuously processes incoming messages and sends LLM-generated replies.

    This function retrieves messages from a global asynchronous queue, generates a reply using the local LLM,
    prints the generated reply for debugging purposes, and sends the reply through the provided Meshtastic interface
    on channel 0. It includes a brief delay between responses to prevent spamming.

    Parameters:
        iface (SerialInterface): The Meshtastic SerialInterface object used to send text messages.

    Side Effects:
        - Consumes messages from the global message_queue.
        - Sends generated text messages over the Meshtastic network using the provided interface.
        - Prints both the incoming messages and generated replies to standard output.

    Returns:
        None
    """
    while True:
        sender, incoming_msg = await message_queue.get()
        reply_msg = await generate_llm_response(incoming_msg)
        print(f"LLM reply to {sender}: {reply_msg}\n")
        iface.sendText(reply_msg, channelIndex=0)
        await asyncio.sleep(0.5)


async def main():
    """
    Main asynchronous entry point for the Serial Meshtastic Chat Client.

    This function sets up the Meshtastic SerialInterface using a serial port provided as a command-line argument,
    subscribes to incoming text messages, and launches the LLM responder task. It also prints connection status
    and instructions for exiting the program.

    Parameters:
        None (the serial port is specified via command-line arguments)

    Side Effects:
        - Connects to a Meshtastic device over the specified serial port.
        - Subscribes to the "meshtastic.receive.text" topic to receive incoming messages.
        - Launches an asynchronous task that automatically responds to incoming messages using the local LLM.
        - Prints status messages and instructions to standard output.
        - Exits gracefully upon receiving a KeyboardInterrupt (Ctrl+C).

    Returns:
        None
    """
    if len(sys.argv) < 2:
        print("Usage: python simcc_llm.py <SERIAL_PORT>")
        sys.exit(1)

    port = sys.argv[1].strip()
    print(f"Attempting to connect to Meshtastic device at serial port: {port}")

    try:
        iface = SerialInterface(devPath=port)
        print("Connected to Meshtastic device over serial!")
        time.sleep(2)  # allow the serial link to stabilize.
    except Exception as e:
        print("Error initializing SerialInterface:", e)
        sys.exit(1)

    print("\nSerial Meshtastic Chat Client (Local LLM) 0.3.0")
    print("-----------------------------------------------")
    print("Auto-reply mode: The LLM will respond to any incoming messages.\n")
    print("Press Ctrl+C to exit...\n")

    # start the LLM responder task
    responder_task = asyncio.create_task(run_llm_responder(iface))

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        responder_task.cancel()
        iface.close()
        sys.exit(0)


if __name__ == "__main__":
    # subscribe to incoming text messages.
    pub.subscribe(onReceive, "meshtastic.receive.text")
    asyncio.run(main())
```

<br>

## License
[Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)

import ctypes
from ctypes import wintypes
import time

if not hasattr(wintypes, "ULONG_PTR"):
    # ULONG_PTR is a pointer-sized unsigned integer (32-bit on x86, 64-bit on x64)
    if ctypes.sizeof(ctypes.c_void_p) == 8:
        wintypes.ULONG_PTR = ctypes.c_uint64
    else:
        wintypes.ULONG_PTR = ctypes.c_uint32

# Input types
INPUT_KEYBOARD = 1

# Event flags
KEYEVENTF_KEYUP       = 0x0002
KEYEVENTF_SCANCODE    = 0x0008 # Use Scan Codes for maximum compatibility (e.g., in games)

# Virtual Key Codes (VK)
VK_CODES = {
    "w": 0x57,
    "a": 0x41,
    "s": 0x53,
    "d": 0x44,
    "f": 0x46,
    "space": 0x20,
    "shift": 0x10
}

# --- Define Structures ---

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk",         wintypes.WORD),
        ("wScan",       wintypes.WORD),
        ("dwFlags",     wintypes.DWORD),
        ("time",        wintypes.DWORD),
        ("dwExtraInfo", wintypes.ULONG_PTR)
    ]

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx",          ctypes.c_long),
        ("dy",          ctypes.c_long),
        ("mouseData",   wintypes.DWORD),
        ("dwFlags",     wintypes.DWORD),
        ("time",        wintypes.DWORD),
        ("dwExtraInfo", wintypes.ULONG_PTR)
    ]

class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = [
            ("ki", KEYBDINPUT),
            ("mi", MOUSEINPUT)
        ]
    _anonymous_ = ("_input",)
    _fields_ = [
        ("type", wintypes.DWORD),
        ("_input", _INPUT)
    ]

# --- Load Windows Functions ---

SendInput = ctypes.windll.user32.SendInput
MapVirtualKeyA = ctypes.windll.user32.MapVirtualKeyA
GetLastError = ctypes.windll.kernel32.GetLastError

# --- Key Press Function ---

def key_down(vk_codes):
    """
    Presses (key down) multiple virtual-key codes at once.
    """
    inputs = []
    for vk in vk_codes:
        scan = MapVirtualKeyA(vk, 0)
        inp = INPUT(type=INPUT_KEYBOARD,
                    ki=KEYBDINPUT(wVk=0, wScan=scan, dwFlags=KEYEVENTF_SCANCODE))
        inputs.append(inp)

    # Send all key-down events at once
    SendInput(len(inputs), (INPUT * len(inputs))(*inputs), ctypes.sizeof(INPUT))


def key_up(vk_codes):
    """
    Releases (key up) multiple virtual-key codes at once.
    """
    inputs = []
    for vk in vk_codes:
        scan = MapVirtualKeyA(vk, 0)
        inp = INPUT(type=INPUT_KEYBOARD,
                    ki=KEYBDINPUT(wVk=0, wScan=scan, dwFlags=KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP))
        inputs.append(inp)

    SendInput(len(inputs), (INPUT * len(inputs))(*inputs), ctypes.sizeof(INPUT))


def press_keys(vk_codes, duration=0.1):
    """
    Press and hold multiple keys, then release.
    """
    key_down(vk_codes)
    time.sleep(duration)
    key_up(vk_codes)
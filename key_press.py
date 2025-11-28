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

def press_key(vk_code, duration=0.1):
    """
    Simulates pressing a key using the robust Scan Code method.
    """
    
    # 1. Map Virtual Key to Scan Code
    # MapVirtualKeyA(VK_code, MAPVK_VK_TO_VSC (0)) returns the scan code
    scan_code = MapVirtualKeyA(vk_code, 0)
    
    # 2. Key Down (Event 1)
    # dwFlags includes KEYEVENTF_SCANCODE
    ii_down = INPUT(type=INPUT_KEYBOARD, 
                    ki=KEYBDINPUT(wVk=0, wScan=scan_code, dwFlags=KEYEVENTF_SCANCODE))
    
    # Send Key Down and check for success
    result = SendInput(1, ctypes.byref(ii_down), ctypes.sizeof(ii_down))
    if result == 0:
        print(f"Key Down FAILED. Last Error: {GetLastError()}")
        return

    # 3. Hold
    if duration > 0:
        time.sleep(duration)
    
    # 4. Key Up (Event 2)
    # dwFlags includes KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP
    ii_up = INPUT(type=INPUT_KEYBOARD, 
                  ki=KEYBDINPUT(wVk=0, wScan=scan_code, dwFlags=KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP))
    
    # Send Key Up
    SendInput(1, ctypes.byref(ii_up), ctypes.sizeof(ii_up))
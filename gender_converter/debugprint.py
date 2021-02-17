import os
import sys
import shutil
import inspect
import logging


def print_verbose(msg="", *msg_kw, verbose=True):
     if verbose:
         print(msg, *msg_kw, file=sys.stderr)
         sys.stderr.flush()


def print_debug(msg="", *msg_kw, stack_level=1, verbose=True):
     """
     prints file and line location of this call and then the msg

     :param msg: (str) message (it is converted to a str if not a str
with str()) (default: "")
     :param stack_level: (int)
     :param verbose: (bool) if set to False, nothing will happen
     """
     if verbose:
         total_stack = inspect.stack()
         frame_info = total_stack[stack_level][0]

         file_name = frame_info.f_code.co_filename
         line_number = frame_info.f_lineno

         fileline = "%s:%d" % (file_name, line_number)

         print_verbose("%-20s %s" % (fileline, str(msg)), *msg_kw)


if __name__ == "__main__":
    print('hello')
    print_debug('hello 2')
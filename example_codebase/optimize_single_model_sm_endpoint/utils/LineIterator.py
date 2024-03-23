# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# --
# --  Author:        Pavan Kumar Rao Navule
# --  Date:          1/12/2023
# --  Purpose:       Iterates over the byte stream from Llama 2 Chat models inferenced with LMI Container with djl-deepspeed version 0.25.0.
# --  Version:       0.1.0
# --  Disclaimer:    This script is provided "as is" in accordance with the repository license
# --  History
# --  When        Version     Who         What
# --  -----------------------------------------------------------------
# --  1/12/2023  0.1.0       Pavan Kumar Rao Navule    Initial
# --  -----------------------------------------------------------------
# --

import io
import re

NEWLINE = re.compile(r'\\n')  
DOUBLE_NEWLINE = re.compile(r'\\n\\n')

class LineIterator:
    """
    A helper class for parsing the byte stream from Llama 2 model inferenced with LMI Container. 
    
    The output of the model will be in the following repetetive but incremental format:
    ```
    b'{"generated_text": "'
    b'lo from L"'
    b'LM \\n\\n'
    b'How are you?"}'
    ...

    For each iteration, we just read the incremental part and seek for the new position for the next iteration till the end of the line.

    """
    
    def __init__(self, stream):
        self.byte_iterator = iter(stream)
        self.buffer = io.BytesIO()
        self.read_pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        start_sequence = b'{"generated_text": "'
        stop_sequence = b'"}'
        new_line = '\n'
        double_new_line = '\n\n'
        while True:
            self.buffer.seek(self.read_pos)
            line = self.buffer.readline()
            if line:
                self.read_pos += len(line)
                if line.startswith(start_sequence):# in :
                    line = line.lstrip(start_sequence)
                
                if line.endswith(stop_sequence):
                    line =line.rstrip(stop_sequence)
                line = line.decode('utf-8')
                line = NEWLINE.sub(new_line, line)
                line = DOUBLE_NEWLINE.sub(double_new_line, line)
                return line
            try:
                chunk = next(self.byte_iterator)
            except StopIteration:
                if self.read_pos < self.buffer.getbuffer().nbytes:
                    continue
                raise
            if 'PayloadPart' not in chunk:
                print('Unknown event type:' + chunk)
                continue
            self.buffer.seek(0, io.SEEK_END)
            self.buffer.write(chunk['PayloadPart']['Bytes'])
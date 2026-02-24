# TanayLabUtilities

Generic utilities used by the Julia code for the Tanay lab.

This is a somewhat arbitrary collection of "generally useful" functions, created on an as-needed bases. No claim is made
for true universality or for suitability for a particular purpose.

See the [v0.1.0 documentation](https://tanaylab.github.io/TanayLabUtilities.jl/v0.1.0) for details.

## Environment variables

The following environment variables affect the code's behavior:

`TLU_FLAME_MEASUREMENTS_FILE` - if set, then coarse flame graph measurements will be appended to the specified file.

`TLU_IS_PATH_CACHE_TIMEOUT_NS` - if set, overrides the 10-second timeout for refreshing the cached entries of which
files do/not exist in a directory. Set to 0 to completely disable caching or to a negative value (-1) when all relevant
file system modifications are done by the program itself (and the code clears the cache when such modifications are
done).

`TLU_LIVE_BYTES_GC_THRESHOLD_FRACTION` - if set, then parallel loops will disable garbage collection as long as the
total memory used by the program is less than this fraction of the machine's total memory. Don't use it unless you know
what you are doing.

## Installation

Just `Pkg.add("TanayLabUtilities")`, like installing any other Julia package.

## License (MIT)

Copyright © 2025 Weizmann Institute of Science

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

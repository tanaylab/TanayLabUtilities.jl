# TanayLabUtilities

Generic utilities used by the Julia code for the Tanay lab.

This is a somewhat arbitrary collection of "generally useful" functions, created on an as-needed bases. No claim is made
for true universality or for suitability for a particular purpose.

See the [v0.1.0 documentation](https://tanaylab.github.io/TanayLabUtilities.jl/v0.1.0) for details.

## Environment variables

The following environment variables affect the code's behavior:

`TLU_FLAME_MEASUREMENTS_FILE` - if set, then coarse flame graph measurements will be appended to the specified file.

`TLU_IS_PATH_CACHE_TIMEOUT_NS` - if set, disables or specifies a timeout for refreshing the cached entries of which
files do/not exist in directories of interest. The default is a negative value (cache indefinitely, which "should" be
fastest and safe). Set to 0 to completely disable caching or to the number of nanoseconds to cache entries for.

If any of the above is set to a value that triggers the behavior, an `@info` log message is generated under the
`tlu_env` group.

## Debug Logging

Two modules generate debug messages:

`ExtendedReadWriteLock` - will generate log messages about read-write locking. This will generate a lot of messages
(since read locks are ubiquitous). This is very rarely useful but when it is needed it is a lifesaver.

`GlobalWeakCache` - will generate messages for every access of the global weak cache. This shouldn't be a lot of data.
Useful for debugging whether global objects (e.g. open `Daf` repositories) are actually cached and reused.

`CachedIsPath` - will generate messages for every operation of the `cached_ispath`. This will generate a lot of messages
(since `Daf` files format uses the file system as a database of sorts). Useful for debugging whether the cache is
properly cleared when file operations are done.

## Installation

Just `Pkg.add("TanayLabUtilities")`, like installing any other Julia package.

## License (MIT)

Copyright © 2025, 2026 Weizmann Institute of Science

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

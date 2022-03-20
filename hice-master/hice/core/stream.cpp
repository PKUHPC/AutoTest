#include "hice/core/stream.h"

namespace hice {

// Not very parseable, but I don't know a good compact syntax for streams.
// Feel free to change this into something more compact if needed.
std::ostream& operator<<(std::ostream& stream, const Stream& s) {
  stream << "stream " << s.id() << " on device " << s.device();
  return stream;
}

} // namespace hice

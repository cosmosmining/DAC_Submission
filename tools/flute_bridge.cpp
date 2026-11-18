// flute_bridge.cpp — C ABI around FLUTE (InstantGR's src/flute.hpp, BSD-3) for ctypes.
//   g++ -O3 -shared -fPIC -std=c++17 -I../external/InstantGR/src flute_bridge.cpp -o libflute_bridge.so
// flute_tree(d, x, y, out_x, out_y, out_n) writes the 2d-2 branches of the rectilinear Steiner
// tree (branch i = point (out_x[i], out_y[i]) joined to branch out_n[i]); returns the count.
#include "flute.hpp"
extern "C" {
int flute_init(const char* powv, const char* post) { flute::readLUT(powv, post); return 0; }
int flute_tree(int d, int* x, int* y, int* ox, int* oy, int* on) {
  flute::Tree t = flute::flute(d, x, y, ACCURACY);
  int m = 2 * t.deg - 2;
  for (int i = 0; i < m; i++) { ox[i] = t.branch[i].x; oy[i] = t.branch[i].y; on[i] = t.branch[i].n; }
  free(t.branch);
  return m;
}
}

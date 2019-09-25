#include "a1.h"
#include <iostream>

using namespace std;

// This is a way for you to test your functions.
// We will only grade the contents of a1.cpp and Image.cpp
int main() {
  cout << "nothing done in a1_main.cpp, debug me !" << endl;

  // Example
  Image im("./Input/castle_small.png");
  std::vector<Image> LC = lumiChromi(im);
  LC[0].write("./Output/castle_luminance.png");
  LC[1].write("./Output/castle_chrominance.png");

  create_special().write("./Output/logo.png");

  Image sat("./Input/skies_and_trees.png");
  gamma_code(sat, 2.0).write("./Output/o.png");
  std::vector<Image> p = gamma_test(sat, 4, 1.8);
  p[0].write("./Output/q.png");
  p[1].write("./Output/g.png");

  
}

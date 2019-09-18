/* -----------------------------------------------------------------
 * File:    Image.cpp
 * Created: 2015-08-29
 * Updated: 2019-08-10
 * -----------------------------------------------------------------
 *
 * The 6.815/6.865 Image Class
 *
 * ---------------------------------------------------------------*/

#include "Image.h"

using namespace std;

// --------- HANDOUT  PS01 ------------------------------

long long Image::number_of_elements() const {
  // --------- HANDOUT  PS01 ------------------------------
  // returns the number of elements in the image.
  // An RGB (3 color channels) image of 100 × 100 pixels has 30000 elements
    long long num = 1;
    for (int i=0; i<DIMS; i++) {
        num *= dim_values[i];
    }
    return num;
}

// -------------- Accessors and Setters -----------------
const float &Image::operator()(int x) const {
  // --------- HANDOUT  PS01 ------------------------------
  // Linear accessor to the image data
    if (x<0 || x>=number_of_elements()) {
        throw OutOfBoundsException();
    }
    return image_data[x];
}

const float &Image::operator()(int x, int y) const {
  // --------- HANDOUT  PS01 ------------------------------
  // Accessor to the image data at channel 0
    if (x<0 || x>=width() || 
        y<0 || y>=height()) {
        throw OutOfBoundsException(); 
    }
    return image_data[x*stride(0) + y*stride(1)];
}

const float &Image::operator()(int x, int y, int z) const {
  // --------- HANDOUT  PS01 ------------------------------
  // Accessor to the image data at channel z
    if (x<0 || x>=width() || 
        y<0 || y>=height()||
        z<0 || z>=channels()) {
        throw OutOfBoundsException(); 
    }
    return image_data[x*stride(0) + y*stride(1) + z*stride(2)];
}

float &Image::operator()(int x) {
  // --------- HANDOUT  PS01 ------------------------------
  // Linear setter to the image data
  if (x<0 || x>=number_of_elements()) {
        throw OutOfBoundsException();
    }
    return image_data[x];
}

float &Image::operator()(int x, int y) {
  // --------- HANDOUT  PS01 ------------------------------
  // Setter to the image data at channel 0
    if (x<0 || x>=width() || 
        y<0 || y>=height()) {
        throw OutOfBoundsException(); 
    }
    return image_data[x*stride(0) + y*stride(1)];
}

float &Image::operator()(int x, int y, int z) {
  // --------- HANDOUT  PS01 ------------------------------
  // Setter to the image data at channel z
    if (x<0 || x>=width() || 
        y<0 || y>=height()||
        z<0 || z>=channels()) {
        throw OutOfBoundsException(); 
    }
    return image_data[x*stride(0) + y*stride(1) + z*stride(2)];
}

void Image::set_color(float r, float g, float b) {
  // --------- HANDOUT  PS01 ------------------------------
  // Set the image pixels to the corresponding values
    switch(DIMS) {
        case 1:
            for (int i=0; i<width(); i++) {
                operator()(i) = r;
            }
            break;
        case 2:
            for (int i=0; i<width(); i++) {
                for (int j=0; j<height(); j++) {
                    operator()(i, j) = r;
                }
            }
            break;
        case 3:
            for (int i=0; i<width(); i++) {
                for (int j=0; j<height(); j++) {
                    operator()(i, j, 0) = r;
                    operator()(i, j, 1) = g;
                    operator()(i, j, 2) = b;
                }
            }
            break;
        default:
            cout << "ERROR" << endl;
            break;
    }
}

void Image::create_rectangle(int xstart, int ystart, int xend, int yend,
                             float r, float g, float b) {
  // --------- HANDOUT  PS01 ------------------------------
  // Set the pixels inside the rectangle to the specified color
    switch(DIMS) {
        case 1:
            for (int i=xstart; i<=xend; i++) {
                operator()(i) = r;
            }
            break;
        case 2:
            for (int i=xstart; i<=xend; i++) {
                for (int j=ystart; j<=yend; j++) {
                    operator()(i, j) = r;
                }
            }
            break;
        case 3:
            for (int i=xstart; i<=xend; i++) {
                for (int j=ystart; j<=yend; j++) {
                    operator()(i, j, 0) = r;
                    operator()(i, j, 1) = g;
                    operator()(i, j, 2) = b;
                }
            }
            break;
        default:
            cout << "ERROR" << endl;
            break;
    }
}

void Image::create_line(int xstart, int ystart, int xend, int yend, float r,
                        float g, float b) {
  // --------- HANDOUT  PS01 ------------------------------
  // Create a line segment with specified color
  throw NotImplementedException(); // change this
}

// ---------------- END of PS01 -------------------------------------

/*********************************************************************
 *                    DO NOT EDIT BELOW THIS LINE                    *
 *********************************************************************/

int Image::debugWriteNumber = 0;

Image::Image(int width, int height, int channels, const std::string &name_) {
  initialize_image_metadata(width, height, channels, name_);
  // Initialize image data
  long long size_of_data = 1;
  for (unsigned int k = 0; k < DIMS; k++) {
    size_of_data *= dim_values[k];
  }
  image_data = std::vector<float>(size_of_data, 0.f);
}

void Image::initialize_image_metadata(int w, int h, int c,
                                      const std::string &name_) {
  if (w < 1)
    throw IllegalDimensionException();
  if (h < 1)
    throw IllegalDimensionException();
  if (c != 1 && c != 3)
    throw IllegalDimensionException();

  dim_values[0] = w;
  dim_values[1] = h;
  dim_values[2] = c;
  stride_[0] = 1;
  stride_[1] = w;
  stride_[2] = w * h;
  image_name = name_;
}

Image::Image(const std::string &filename) {
  std::vector<unsigned char> uint8_image;
  unsigned int h;
  unsigned int w;
  unsigned int c = 4;
  unsigned int oc = 3; // Throw away transparency

  // Decode PNG file In column major order with packed color values
  unsigned err = lodepng::decode(uint8_image, w, h, filename.c_str());
  if (err == 48) {
    throw FileNotFoundException();
  }

  image_data = std::vector<float>(h * w * oc, 0);

  for (unsigned int x = 0; x < w; x++) {
    for (unsigned int y = 0; y < h; y++) {
      for (unsigned int z = 0; z < oc; z++) {
        image_data[x + y * w + z * w * h] =
            uint8_to_float(uint8_image[z + x * c + y * c * w]);
      }
    }
  }

  initialize_image_metadata(w, h, oc, filename);
}

Image::~Image() {} // Nothing to clean up

void Image::write(const std::string &filename) const {
  if (channels() != 1 && channels() != 3 && channels() != 4)
    throw ChannelException();
  int png_channels = 4;
  std::vector<unsigned char> uint8_image(height() * width() * png_channels,
                                         255);
  int z;
  for (int x = 0; x < width(); x++) {
    for (int y = 0; y < height(); y++) {
      for (z = 0; z < channels(); z++) {
        uint8_image[z + x * png_channels + y * png_channels * width()] =
            float_to_uint8(
                image_data[x + y * width() + z * width() * height()]);
      }
      for (; z < 3; z++) { // Only executes when there is one channel
        uint8_image[z + x * png_channels + y * png_channels * width()] =
            float_to_uint8(
                image_data[x + y * width() + 0 * width() * height()]);
      }
    }
  }
  lodepng::encode(filename.c_str(), uint8_image, width(), height());
}

void Image::debug_write() const {
  std::ostringstream ss;
  ss << "./Output/" << debugWriteNumber << ".png";
  std::string filename = ss.str();
  write(filename);
  debugWriteNumber++;
}

float Image::uint8_to_float(const unsigned char &in) {
  return ((float)in) / (255.0f);
}

unsigned char Image::float_to_uint8(const float &in) {
  float out = in;
  if (out < 0)
    out = 0;
  if (out > 1)
    out = 1;
  return (unsigned char)(255.0f * out);
}

void compareDimensions(const Image &im1, const Image &im2) {
  for (int i = 0; i < 3; i++) {
    if (im1.extent(i) != im2.extent(i))
      throw MismatchedDimensionsException();
  }
}

Image operator+(const Image &im1, const Image &im2) {
  compareDimensions(im1, im2);
  long long total_pixels = im1.number_of_elements();

  Image output(im1.extent(0), im1.extent(1), im1.extent(2));
  for (int i = 0; i < total_pixels; i++) {
    output(i) = im1(i) + im2(i);
  }
  return output;
}

Image operator-(const Image &im1, const Image &im2) {
  compareDimensions(im1, im2);
  long long total_pixels = im1.number_of_elements();
  Image output(im1.extent(0), im1.extent(1), im1.extent(2));
  for (int i = 0; i < total_pixels; i++) {
    output(i) = im1(i) - im2(i);
  }
  return output;
}

Image operator*(const Image &im1, const Image &im2) {
  compareDimensions(im1, im2);
  long long total_pixels = im1.number_of_elements();
  Image output(im1.extent(0), im1.extent(1), im1.extent(2));
  for (int i = 0; i < total_pixels; i++) {
    output(i) = im1(i) * im2(i);
  }
  return output;
}

Image operator/(const Image &im1, const Image &im2) {
  compareDimensions(im1, im2);
  long long total_pixels = im1.number_of_elements();
  Image output(im1.extent(0), im1.extent(1), im1.extent(2));
  for (int i = 0; i < total_pixels; i++) {
    if (im2(i) == 0)
      throw DivideByZeroException();
    output(i) = im1(i) / im2(i);
  }
  return output;
}

Image operator+(const Image &im1, const float &c) {
  long long total_pixels = im1.number_of_elements();
  Image output(im1.extent(0), im1.extent(1), im1.extent(2));
  for (int i = 0; i < total_pixels; i++) {
    output(i) = im1(i) + c;
  }
  return output;
}

Image operator-(const Image &im1, const float &c) {
  long long total_pixels = im1.number_of_elements();
  Image output(im1.extent(0), im1.extent(1), im1.extent(2));
  for (int i = 0; i < total_pixels; i++) {
    output(i) = im1(i) - c;
  }
  return output;
}
Image operator*(const Image &im1, const float &c) {
  long long total_pixels = im1.number_of_elements();
  Image output(im1.extent(0), im1.extent(1), im1.extent(2));
  for (int i = 0; i < total_pixels; i++) {
    output(i) = im1(i) * c;
  }
  return output;
}
Image operator/(const Image &im1, const float &c) {
  long long total_pixels = im1.number_of_elements();
  Image output(im1.extent(0), im1.extent(1), im1.extent(2));
  if (c == 0)
    throw DivideByZeroException();
  for (int i = 0; i < total_pixels; i++) {
    output(i) = im1(i) / c;
  }
  return output;
}

Image operator+(const float &c, const Image &im1) {
  long long total_pixels = im1.number_of_elements();
  Image output(im1.extent(0), im1.extent(1), im1.extent(2));
  for (int i = 0; i < total_pixels; i++) {
    output(i) = im1(i) + c;
  }
  return output;
}

Image operator-(const float &c, const Image &im1) {
  long long total_pixels = im1.number_of_elements();
  Image output(im1.extent(0), im1.extent(1), im1.extent(2));
  for (int i = 0; i < total_pixels; i++) {
    output(i) = c - im1(i);
  }
  return output;
}

Image operator*(const float &c, const Image &im1) {
  long long total_pixels = im1.number_of_elements();
  Image output(im1.extent(0), im1.extent(1), im1.extent(2));
  for (int i = 0; i < total_pixels; i++) {
    output(i) = im1(i) * c;
  }
  return output;
}
Image operator/(const float &c, const Image &im1) {
  long long total_pixels = im1.number_of_elements();
  Image output(im1.extent(0), im1.extent(1), im1.extent(2));
  for (int i = 0; i < total_pixels; i++) {
    if (im1(i) == 0)
      throw DivideByZeroException();
    output(i) = c / im1(i);
  }
  return output;
}

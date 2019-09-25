/* -----------------------------------------------------------------
 * File:    filtering.cpp
 * Created: 2015-09-22
 * -----------------------------------------------------------------
 *
 * Image convolution and filtering
 *
 * ---------------------------------------------------------------*/

#include "filtering.h"
#include <cassert>
#include <cmath>

using namespace std;

Image boxBlur(const Image &im, int k, bool clamp) {
  // --------- HANDOUT  PS02 ------------------------------
  // Convolve an image with a box filter of size k by k
  // It is safe to asssume k is odd.
  Image output(im.width(), im.height(), im.channels());
  for (int i = 0; i < im.width(); i++) {
    for (int j = 0; j < im.height(); j++) {
      for (int k = 0; k < im.channels(); k++) {
        float a = 0.0f;
        for (int x = -(k-1)/2; x <= (k-1)/2; x++) {
          for (int y = -(k-1)/2; y <= (k-1)/2; y++) {
            a += im.smartAccessor(i + x, j + y, k, clamp);
          }
        }
        output(i, j, k) = a / (2 * (k-1)/2 + 1) / (2 * (k-1)/2 + 1);
      }
    }
  }
  return output;
}

Image Filter::convolve(const Image &im, bool clamp) {
  // --------- HANDOUT  PS02 ------------------------------
  // Write a convolution function for the filter class
  Image output(im.width(), im.height(), im.channels());
  for (int i = 0; i < im.width(); i++) {
    for (int j = 0; j < im.height(); j++) {
      for (int k = 0; k < im.channels(); k++) {
        float p = 0.0f;
        for (int x = 0; x < width; x++) {
          for (int y = 0; y < height; y++) {
            int w = (this->width-1)/2;
            int h = (this->height-1)/2;
            float temp = im.smartAccessor(i + x - w, j + y - h, k, clamp);
            p += temp * (*this)(width - 1 - x, height - 1 - y);
          }
        }
        output(i, j, k) = p;
      }
    }
  }
  return output;
}

Image boxBlur_filterClass(const Image &im, int k, bool clamp) {
  // --------- HANDOUT  PS02 ------------------------------
  // Reimplement the box filter using the filter class.
  // check that your results match those in the previous function "boxBlur"
  vector<float> kernel;
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < k; j++) {
      kernel.push_back(1 / k / k);
    }
  }
  Filter f(kernel, k, k);
  return f.convolve(im, clamp);
}

Image gradientMagnitude(const Image &im, bool clamp) {
  // --------- HANDOUT  PS02 ------------------------------
  // Uses a Sobel kernel to compute the horizontal and vertical components
  // of the gradient of an image and returns the gradient magnitude.
  vector<float> x{-1, 0, 1,
                      -2, 0, 2,
                      -1, 0, 1};
  
  vector<float> y{-1,-2,-1,
                      0, 0, 0,
                      1, 2, 1};

  Filter xf(x, 3, 3);
  Filter yf(y, 3, 3);

  Image xc = xf.convolve(im, clamp);
  Image yc = yf.convolve(im, clamp);

  Image output(im.width(), im.height(), im.channels());
  output = xc * xc + yc * yc;
  for (int i = 0; i < im.width(); i++) {
    for (int j = 0; j < im.height(); j++) {
      for (int k = 0; k < im.channels(); k++) {
        output(i, j, k) = sqrt(output(i, j, k));
      }
    }
  }
  return output;
}

vector<float> gauss1DFilterValues(float sigma, float truncate) {
  // --------- HANDOUT  PS02 ------------------------------
  // Create a vector containing the normalized values in a 1D Gaussian filter
  // Truncate the gaussian at truncate*sigma.
  vector<float> filter;
  float t = 0.0f;
  for(int i = -ceil(sigma * truncate); i <= ceil(sigma * truncate); i++) {
    t += exp(-i * i / 2.0 / sigma / sigma);
  }
  for(int i = -ceil(sigma * truncate); i <= ceil(sigma * truncate); i++) {
    filter.push_back(exp(-i * i / 2.0 / sigma / sigma) / t);
  }
  return filter;
}

Image gaussianBlur_horizontal(const Image &im, float sigma, float truncate,
                              bool clamp) {
  // --------- HANDOUT  PS02 ------------------------------
  // Gaussian blur across the rows of an image
  Filter tmp(gauss1DFilterValues(sigma, truncate), 1 + 2 * ceil(sigma * truncate), 1);
  return tmp.convolve(im, clamp);
}

vector<float> gauss2DFilterValues(float sigma, float truncate) {
  // --------- HANDOUT  PS02 ------------------------------
  // Create a vector containing the normalized values in a 2D Gaussian
  // filter. Truncate the gaussian at truncate*sigma.
  vector<float> filter;
  float t = 0.0f;
  for(int i = -ceil(sigma * truncate); i <= ceil(sigma * truncate); i++) {
    for (int j = -ceil(sigma * truncate); j <= ceil(sigma * truncate); j++) {
      t += exp(-(i * i + j * j) / 2.0 / sigma / sigma);
    }
  }
  for(int i = -ceil(sigma * truncate); i <= ceil(sigma * truncate); i++) {
    for (int j = -ceil(sigma * truncate); j <= ceil(sigma * truncate); j++) {
      filter.push_back(exp(-(i * i + j * j) / 2.0 / sigma / sigma) / t);
    }
  }
  return filter;
}

Image gaussianBlur_2D(const Image &im, float sigma, float truncate,
                      bool clamp) {
  // --------- HANDOUT  PS02 ------------------------------
  // Blur an image with a full 2D rotationally symmetric Gaussian kernel
  Filter x(gauss2DFilterValues(sigma, truncate),
             1 + 2 * ceil(sigma * truncate),
             1 + 2 * ceil(sigma * truncate));
  return x.convolve(im, clamp);
}

Image gaussianBlur_separable(const Image &im, float sigma, float truncate,
                             bool clamp) {
  // --------- HANDOUT  PS02 ------------------------------
  // Use principles of seperabiltity to blur an image using 2 1D Gaussian
  // Filters
  Filter v(gauss1DFilterValues(sigma, truncate), 1, 1 + 2 * ceil(sigma * truncate));
  Filter h(gauss1DFilterValues(sigma, truncate), 1 + 2 * ceil(sigma * truncate), 1);
  return v.convolve(h.convolve(im, clamp), clamp);
}

Image unsharpMask(const Image &im, float sigma, float truncate, float strength,
                  bool clamp) {
  // --------- HANDOUT  PS02 ------------------------------
  // Sharpen an image
  Image lp = gaussianBlur_separable(im, sigma, truncate, clamp);
  return strength * (im - lp) + im;
}

Image bilateral(const Image &im, float sigmaRange, float sigmaDomain,
                float truncateDomain, bool clamp) {
  // --------- HANDOUT  PS02 ------------------------------
  // Denoise an image using the bilateral filter
  Image output(im.width(), im.height(), im.channels());

  vector<float> sd = gauss1DFilterValues(sigmaDomain, truncateDomain);
  
  for (int i = 0; i < im.width(); i++) {
    for (int j = 0; j < im.height(); j++) {
      for (int k = 0; k < im.channels(); k++) {
        float w = 0.0f;
        float p = 0.0f;
        for (int x = -ceil(sigmaDomain * truncateDomain); x <= ceil(sigmaDomain * truncateDomain); x++) {
          for (int y = -ceil(sigmaDomain * truncateDomain); y <= ceil(sigmaDomain * truncateDomain); y++) {
            float diff = pow(im(i, j, 0) - im.smartAccessor(i + x, j + y, 0, clamp), 2) +
                              pow(im(i, j, 1) - im.smartAccessor(i + x, j + y, 1, clamp), 2) +
                              pow(im(i, j, 2) - im.smartAccessor(i + x, j + y, 2, clamp), 2);
                              
            float weight = sd[ceil(sigmaDomain * truncateDomain) + x] 
                            * sd[ceil(sigmaDomain * truncateDomain) + y]
                            * exp(-diff / 2.0 / sigmaRange / sigmaRange);
            w += weight;
            p += weight * im.smartAccessor(i + x, j + y, k, clamp);
          }
        }
        output(i, j, k) = p / w;
      }
    }
  }
  return output;
}

Image bilaYUV(const Image &im, float sigmaRange, float sigmaY, float sigmaUV,
              float truncateDomain, bool clamp) {
  // --------- HANDOUT  PS02 ------------------------------
  // 6.865 only
  // Bilaterial Filter an image seperatly for
  // the Y and UV components of an image
  return im;
}

/**************************************************************
 //               DON'T EDIT BELOW THIS LINE                //
 *************************************************************/

// Create an image of 0's with a value of 1 in the middle. This function
// can be used to test that you have properly set the kernel values in your
// Filter object. Make sure to set k to be larger than the size of your kernel
Image impulseImg(int k) {
  // initlize a kxkx1 image of all 0's
  Image impulse(k, k, 1);

  // set the center pixel to have intensity 1
  int center = floor(k / 2);
  impulse(center, center, 0) = 1.0f;

  return impulse;
}

// ------------- FILTER CLASS -----------------------
Filter::Filter(const vector<float> &fData, int fWidth, int fHeight)
    : kernel(fData), width(fWidth), height(fHeight) {
  assert(fWidth * fHeight == (int)fData.size());
}

Filter::Filter(int fWidth, int fHeight)
    : kernel(std::vector<float>(fWidth * fHeight, 0)), width(fWidth),
      height(fHeight) {}

Filter::~Filter() {}

const float &Filter::operator()(int x, int y) const {
  if (x < 0 || x >= width)
    throw OutOfBoundsException();
  if (y < 0 || y >= height)
    throw OutOfBoundsException();

  return kernel[x + y * width];
}

float &Filter::operator()(int x, int y) {
  if (x < 0 || x >= width)
    throw OutOfBoundsException();
  if (y < 0 || y >= height)
    throw OutOfBoundsException();

  return kernel[x + y * width];
}
// --------- END FILTER CLASS -----------------------

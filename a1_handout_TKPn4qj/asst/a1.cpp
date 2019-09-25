/* -----------------------------------------------------------------
 * File:    a1.cpp
 * Created: 2015-09-15
 * Updated: 2019-08-10
 * -----------------------------------------------------------------
 *
 * Assignment 01
 *
 * ---------------------------------------------------------------*/

#include "a1.h"
#include <cmath>
using namespace std;

// Create a surprise image
Image create_special() {
  // // --------- HANDOUT  PS01 ------------------------------
  // create the image outlined in the handout
  Image im(290, 150, 3);
  im.set_color(1.0, 1.0, 1.0);
  im.create_rectangle(0, 0, 31, 149, .64, .12, .20);
  im.create_rectangle(52, 0, 83, 102, .64, .12, .20);
  im.create_rectangle(103, 0, 134, 149, .64, .12, .20);
  im.create_rectangle(155, 0, 186, 30, .64, .12, .20);
  im.create_rectangle(155, 48, 186, 149, .55, .55, .55);
  im.create_rectangle(207, 0, 289, 30, .64, .12, .20);
  im.create_rectangle(207, 48, 238, 149, .64, .12, .20);
  return im;
}

// Change the brightness of the image
// const Image & means a reference to im will get passed to the function,
// but the compiler won't let you modify it within the function.
// So you will return a new image
Image brightness(const Image &im, float factor) {
  // --------- HANDOUT  PS01 ------------------------------
  Image output(im.width(), im.height(), im.channels());
  // Modify image brightness
  // return output;
  for (int i=0; i<im.width(); i++) {
    for (int j=0; j<im.height(); j++) {
      for (int k=0; k<im.channels(); k++) {
        float x = factor * im(i, j, k);
        x = max(0.0f, min(1.0f, x));
        output(i, j, k) = x;
      }
    }
  }
  return output;
}

Image contrast(const Image &im, float factor, float midpoint) {
  // --------- HANDOUT  PS01 ------------------------------
  Image output(im.width(), im.height(), im.channels());
  // Modify image contrast
  // return output;
  for (int i=0; i<im.width(); i++) {
    for (int j=0; j<im.height(); j++) {
      for (int k=0; k<im.channels(); k++) {
        float x = factor * (im(i, j, k) - midpoint);
        x += midpoint;
        x = min(1.0f, x);
        x = max(0.0f, x);
        output(i, j, k) = x;
      }
    }
  }
  return output;
}

Image color2gray(const Image &im, const std::vector<float> &weights) {
  // --------- HANDOUT  PS01 ------------------------------
  Image output(im.width(), im.height(), 1);
  // Convert to grayscale
  for (int i=0; i<im.width(); i++) {
    for (int j=0; j<im.height(); j++) {
      float total = 0.0f;
      for (int k=0; k < im.channels(); k++) {
        total += weights[k] * im(i, j, k);
      }
      float total_weights = weights[0] + weights[1] + weights[2];
      output(i, j) = total / total_weights;
    }
  }
  return output;
}

// For this function, we want two outputs, a single channel luminance image
// and a three channel chrominance image. Return them in a vector with
// luminance first
std::vector<Image> lumiChromi(const Image &im) {
  // --------- HANDOUT  PS01 ------------------------------
  // Create the luminance image
  Image lum = color2gray(im);
  // Create the chrominance image
  Image chrom(im.width(), im.height(), 3);
  for (int i=0; i<im.width(); i++) {
    for (int j=0; j<im.height(); j++) {
      for (int k=0; k<im.channels(); k++) {
        if (lum(i, j) != 0) {
            chrom(i, j, k) = im(i, j, k) / lum(i, j);
        } 
        else {
            chrom(i, j, k) = 0.0;
        }
          
      }
    }
  }
  // Create the output vector as (luminance, chrominance)
  std::vector<Image> out_vec;
  out_vec.push_back(lum);
  out_vec.push_back(chrom);
	return out_vec;
}

// Modify brightness then contrast
Image brightnessContrastLumi(const Image &im, float brightF, float contrastF,
                             float midpoint) {
  // --------- HANDOUT  PS01 ------------------------------
  // Modify brightness, then contrast of luminance image
  std::vector<Image> lumiChrom = lumiChromi(im);
  Image newLumi = contrast(brightness(lumiChrom[0], brightF), contrastF, midpoint);
  Image output(im.width(), im.height(), im.channels());
  for (int i=0; i<im.width(); i++) {
    for (int j=0; j<im.height(); j++) {
      for (int k=0; k<im.channels(); k++) {
        output(i, j, k) = newLumi(i, j) * lumiChrom[1](i, j, k);
      }
    }
  }
  return output;
}

Image rgb2yuv(const Image &im) {
  // --------- HANDOUT  PS01 ------------------------------
  // Create output image of appropriate size
  // Change colorspace
  Image yuv(im.width(), im.height(), im.channels());
  for (int i=0; i<im.width(); i++) {
    for (int j=0; j<im.height(); j++) {
      yuv(i, j, 0) = (.299 * im(i, j, 0)) + (.587 * im(i, j, 1)) + (.144 * im(i, j, 2));
      yuv(i, j, 1) = (-.147 * im(i, j, 0)) + (-0.289 * im(i, j, 1)) + (.436 * im(i, j, 2));
      yuv(i, j, 2) = (.615 * im(i, j, 0)) + (-.515 * im(i, j, 1)) + (-.100 * im(i, j, 2));
    }
  }
  return yuv;
}

Image yuv2rgb(const Image &im) {
  // --------- HANDOUT  PS01 ------------------------------
  // Create output image of appropriate size
  // Change colorspace
  Image rgb(im.width(), im.height(), im.channels());
  for (int i=0; i<im.width(); i++) {
    for (int j=0; j<im.height(); j++) {
      rgb(i, j, 0) = im(i, j, 0) + 1.14 * im(i, j, 2);
      rgb(i, j, 1) = im(i, j, 0) + -0.395 * im(i, j, 1) + -.581 * im(i, j, 2);
      rgb(i, j, 2) = im(i, j, 0) + 2.032 * im(i, j, 1) + 0;
    }
  }
  return rgb;
}

Image saturate(const Image &im, float factor) {
  // --------- HANDOUT  PS01 ------------------------------
  // Create output image of appropriate size
  // Saturate image
  // return output;
  Image output = rgb2yuv(im);
    for (int i=0; i<im.width(); i++) {
      for (int j=0; j<im.height(); j++) {
        output(i, j, 1) *= factor;
        output(i, j, 2) *= factor;
      }
    }
    return rgb2yuv(output);
}

// Gamma codes the image
Image gamma_code(const Image &im, float gamma) {
  // // --------- HANDOUT  PS01 ------------------------------
  Image output(im.width(), im.height(), im.channels());
  // Gamma encodes the image
  // return output;
  for (int i=0; i<im.width(); i++) {
    for (int j=0; j<im.height(); j++) {
      for (int k=0; k<im.channels(); k++) {
        float temp = 1.0f / gamma;
        output(i, j, k) = pow(im(i, j, k), temp);
      }
    }
  }
  return output;
}

// Quantizes the image to 2^bits levels and scales back to 0~1
Image quantize(const Image &im, int bits) {
  // // --------- HANDOUT  PS01 ------------------------------
  Image output(im.width(), im.height(), im.channels());
  // Quantizes the image to 2^bits levels
  // return output;
  for (int i=0; i<im.width(); i++) {
    for (int j=0; j<im.height(); j++) {
      for (int k=0; k<im.channels(); k++) {
          output(i, j, k) = (round(im(i, j, k) * pow(2, bits))) / 
                            ((float) pow(2, bits));
      }
    }
  }
  return output;
}

// Compare between first quantize then gamma_encode and first gamma_encode
// then quantize
std::vector<Image> gamma_test(const Image &im, int bits, float gamma) {
  // // --------- HANDOUT  PS01 ------------------------------
  // // im1 = quantize then gamma_encode the image
  // // im2 = gamma_encode then quantize the image
  // // Remember to create the output images and the output vector
  // // Push the images onto the vector
  // // Do all the required processing
  // // Return the vector, color image first
  std::vector<Image> output;
  Image im1 = gamma_code(quantize(im, bits), gamma);
  output.push_back(im1);
  Image im2 = quantize(gamma_code(im, gamma), bits);
  output.push_back(im2);
  return output;
}

// Return two images in a C++ vector
std::vector<Image> spanish(const Image &im) {
  // --------- HANDOUT  PS01 ------------------------------
  // Remember to create the output images and the output vector
  // Push the images onto the vector
  // Do all the required processing
  // Return the vector, color image first
  Image yuv = rgb2yuv(im);
  Image first(im.width(), im.height(), im.channels());
  Image second = color2gray(im);
  for (int i=0; i<im.width(); i++) {
    for (int j=0; j<im.height(); j++) {
      first(i, j, 0) = 0.5;
      first(i, j, 1) = -yuv(i, j, 1);
      first(i, j, 2) = -yuv(i, j, 2);
    }
  }
  first = yuv2rgb(first);

  int w = floor(im.width()/2);
  int h = floor(im.height()/2);
  for (int k=0; k<im.channels(); k++) {
    first(w, h, k) = 0;
  }
  second(w, h) = 0;

  std::vector<Image> output;
  output.push_back(first);
  output.push_back(second);
  return output;
}

float avg_channel(const Image &im, int channel) {
  float total = 0.0f;
  for (int i=0; i<im.width(); i++) {
    for (int j=0; j<im.height(); j++) {
        total += im(i, j, channel);
    }
  }
  float x = im.width() / im.height();
  return total / x;
}

// White balances an image using the gray world assumption
Image grayworld(const Image &im) {
  // --------- HANDOUT  PS01 ------------------------------
  // Implement automatic white balance by multiplying each channel
  // of the input by a factor such that the three channel of the output
  // image have the same mean value. The mean value of the green channel
  // is taken as reference.
  float r = avg_channel(im, 0);
  float g = avg_channel(im, 1);
  float rf = g / r;
  float b = avg_channel(im, 2);
  float bf = g / b;

  Image output(im.width(), im.height(), im.channels());

  for (int i=0; i<im.width(); i++) {
    for (int j=0; j<im.height(); j++) {
      output(i, j, 0) = rf * im(i, j, 0);
      output(i, j, 1) = im(i, j, 1);
      output(i, j, 2) = bf * im(i, j, 2);
    }
  }
  return output;
}

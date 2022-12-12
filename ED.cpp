#include "ED.h"
#include <fstream>
#include "EDColor.h"

using namespace cv;
using namespace std;

ED::ED(Mat _srcImage, GradientOperator _op, int _gradThresh, int _anchorThresh, int _scanInterval,
       int _minPathLen, double _sigma, bool _sumFlag)
{
  // Check parameters for sanity
  if (_gradThresh < 1) _gradThresh = 1;
  if (_anchorThresh < 0) _anchorThresh = 0;
  if (_sigma < 1.0) _sigma = 1.0;

  srcImage = _srcImage;

  height = srcImage.rows;
  width = srcImage.cols;

  op = _op;
  gradThresh = _gradThresh;
  anchorThresh = _anchorThresh;
  scanInterval = _scanInterval;
  minPathLen = _minPathLen;
  sigma = _sigma;
  sumFlag = _sumFlag;

  segmentNos = 0;
  segmentPoints.push_back(vector<Point>());  // create empty vector of points for segments

  edgeImage = Mat(height, width, CV_8UC1, Scalar(0));  // initialize edge Image
  smoothImage = Mat(height, width, CV_8UC1);
  dirImage = Mat(height, width, CV_8UC1);
  gradImage = Mat(height, width, CV_16SC1);  // gradImage contains short values

  srcImg = srcImage.data;

  //// Detect Edges By Edge Drawing Algorithm  ////

  /*------------ SMOOTH THE IMAGE BY A GAUSSIAN KERNEL -------------------*/
  if (sigma == 1.0)
    GaussianBlur(srcImage, smoothImage, Size(5, 5), sigma);
  else
    GaussianBlur(srcImage, smoothImage, Size(), sigma);  // calculate kernel from sigma

  // Assign Pointers from Mat's data
  smoothImg = smoothImage.data;
  dirImg = dirImage.data;
  gradImg = (short *)gradImage.data;
  edgeImg = edgeImage.data;

  /*------------ COMPUTE GRADIENT & EDGE DIRECTION MAPS -------------------*/
  ComputeGradient();

  /*------------ COMPUTE ANCHORS -------------------*/
  ComputeAnchorPoints();

  /*------------ JOIN ANCHORS -------------------*/
  JoinAnchorPointsUsingSortedAnchors();
}

// This constructor for use of EDLines and EDCircle with ED given as constructor argument
// only the necessary attributes are coppied
ED::ED(const ED &cpyObj)
{
  height = cpyObj.height;
  width = cpyObj.width;

  srcImage = cpyObj.srcImage.clone();

  op = cpyObj.op;
  gradThresh = cpyObj.gradThresh;
  anchorThresh = cpyObj.anchorThresh;
  scanInterval = cpyObj.scanInterval;
  minPathLen = cpyObj.minPathLen;
  sigma = cpyObj.sigma;
  sumFlag = cpyObj.sumFlag;

  edgeImage = cpyObj.edgeImage.clone();
  smoothImage = cpyObj.smoothImage.clone();
  dirImage = cpyObj.dirImage.clone();
  gradImage = cpyObj.gradImage.clone();

  srcImg = srcImage.data;

  smoothImg = smoothImage.data;
  dirImg = dirImage.data;
  gradImg = (short *)gradImage.data;
  edgeImg = edgeImage.data;

  segmentPoints = cpyObj.segmentPoints;
  segmentNos = cpyObj.segmentNos;
}

// This constructor for use of EDColor with use of direction and gradient image
// It finds edge image for given gradient and direction image
ED::ED(short *_gradImg, uchar *_dirImg, int _width, int _height, int _gradThresh, int _anchorThresh,
       int _scanInterval, int _minPathLen, bool selectStableAnchors)
{
  height = _height;
  width = _width;

  gradThresh = _gradThresh;
  anchorThresh = _anchorThresh;
  scanInterval = _scanInterval;
  minPathLen = _minPathLen;

  gradImg = _gradImg;
  dirImg = _dirImg;

  edgeImage = Mat(height, width, CV_8UC1, Scalar(0));  // initialize edge Image

  edgeImg = edgeImage.data;

  if (selectStableAnchors)
  {
    // Compute anchors with the user supplied parameters
    anchorThresh = 0;  // anchorThresh used as zero while computing anchor points if
                       // selectStableAnchors set. Finding higher number of anchors is OK, because
                       // we have following validation steps in selectStableAnchors.
    ComputeAnchorPoints();
    anchorThresh =
        _anchorThresh;     // set it to its initial argument value for further anchor validation.
    anchorPoints.clear();  // considering validation step below, it should constructed again.

    for (int i = 1; i < height - 1; i++)
    {
      for (int j = 1; j < width - 1; j++)
      {
        if (edgeImg[i * width + j] != ANCHOR_PIXEL) continue;

        // Take only "stable" anchors
        // 0 degree edge
        if (edgeImg[i * width + j - 1] && edgeImg[i * width + j + 1])
        {
          int diff1 = gradImg[i * width + j] - gradImg[(i - 1) * width + j];
          int diff2 = gradImg[i * width + j] - gradImg[(i + 1) * width + j];
          if (diff1 >= anchorThresh && diff2 >= anchorThresh) edgeImg[i * width + j] = 255;

          continue;
        }  // end-if

        // 90 degree edge
        if (edgeImg[(i - 1) * width + j] && edgeImg[(i + 1) * width + j])
        {
          int diff1 = gradImg[i * width + j] - gradImg[i * width + j - 1];
          int diff2 = gradImg[i * width + j] - gradImg[i * width + j + 1];
          if (diff1 >= anchorThresh && diff2 >= anchorThresh) edgeImg[i * width + j] = 255;

          continue;
        }  // end-if

        // 135 degree diagonal
        if (edgeImg[(i - 1) * width + j - 1] && edgeImg[(i + 1) * width + j + 1])
        {
          int diff1 = gradImg[i * width + j] - gradImg[(i - 1) * width + j + 1];
          int diff2 = gradImg[i * width + j] - gradImg[(i + 1) * width + j - 1];
          if (diff1 >= anchorThresh && diff2 >= anchorThresh) edgeImg[i * width + j] = 255;
          continue;
        }  // end-if

        // 45 degree diagonal
        if (edgeImg[(i - 1) * width + j + 1] && edgeImg[(i + 1) * width + j - 1])
        {
          int diff1 = gradImg[i * width + j] - gradImg[(i - 1) * width + j - 1];
          int diff2 = gradImg[i * width + j] - gradImg[(i + 1) * width + j + 1];
          if (diff1 >= anchorThresh && diff2 >= anchorThresh) edgeImg[i * width + j] = 255;
        }  // end-if

      }  // end-for
    }    // end-for

    for (int i = 0; i < width * height; i++)
      if (edgeImg[i] == ANCHOR_PIXEL)
        edgeImg[i] = 0;
      else if (edgeImg[i] == 255)
      {
        edgeImg[i] = ANCHOR_PIXEL;
        int y = i / width;
        int x = i % width;
        anchorPoints.push_back(Point(x, y));  // push validated anchor point to vector
      }

    anchorNos = anchorPoints.size();  // get # of anchor pixels
  }

  else
  {
    // Compute anchors with the user supplied parameters
    ComputeAnchorPoints();  // anchorThresh used as given as argument. No validation applied. (No
                            // stable anchors.)
  }                         // end-else

  segmentNos = 0;
  segmentPoints.push_back(vector<Point>());  // create empty vector of points for segments

  JoinAnchorPointsUsingSortedAnchors();
}

ED::ED(EDColor &obj)
{
  width = obj.getWidth();
  height = obj.getHeight();
  segmentPoints = obj.getSegments();
  segmentNos = obj.getSegmentNo();
}

ED::ED()
{
  //
}

Mat ED::getEdgeImage() { return edgeImage; }

Mat ED::getAnchorImage()
{
  Mat anchorImage = Mat(edgeImage.size(), edgeImage.type(), Scalar(0));

  std::vector<Point>::iterator it;

  for (it = anchorPoints.begin(); it != anchorPoints.end(); it++) anchorImage.at<uchar>(*it) = 255;

  return anchorImage;
}

Mat ED::getSmoothImage() { return smoothImage; }

Mat ED::getGradImage()
{
  Mat result8UC1;
  convertScaleAbs(gradImage, result8UC1);

  return result8UC1;
}

int ED::getSegmentNo() { return segmentNos; }

int ED::getAnchorNo() { return anchorNos; }

std::vector<Point> ED::getAnchorPoints() { return anchorPoints; }

std::vector<std::vector<Point>> ED::getSegments() { return segmentPoints; }

std::vector<std::vector<Point>> ED::getSortedSegments()
{
  // sort segments from largest to smallest
  std::sort(
      segmentPoints.begin(), segmentPoints.end(),
      [](const std::vector<Point> &a, const std::vector<Point> &b) { return a.size() > b.size(); });

  return segmentPoints;
}

Mat ED::drawParticularSegments(std::vector<int> list)
{
  Mat segmentsImage = Mat(edgeImage.size(), edgeImage.type(), Scalar(0));

  std::vector<Point>::iterator it;
  std::vector<int>::iterator itInt;

  for (itInt = list.begin(); itInt != list.end(); itInt++)
    for (it = segmentPoints[*itInt].begin(); it != segmentPoints[*itInt].end(); it++)
      segmentsImage.at<uchar>(*it) = 255;

  return segmentsImage;
}

void ED::ComputeGradient()
{
  cv::Mat kernel;
  cv::Point anchor;
  switch (op)
  {
  case PREWITT_OPERATOR:
    kernel = (cv::Mat_<int>(3, 3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);
    anchor = cv::Point(-1, -1);
    break;
  case SOBEL_OPERATOR:
    kernel = (cv::Mat_<int>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
    anchor = cv::Point(-1, -1);
    break;
  case SCHARR_OPERATOR:
    kernel = (cv::Mat_<int>(3, 3) << -3, -10, -3, 0, 0, 0, 3, 10, 3);
    anchor = cv::Point(-1, -1);
    break;
  case LSD_OPERATOR:
    kernel = (cv::Mat_<int>(2, 2) << -1, -1, 1, 1);
    anchor = cv::Point(0, 0);
    break;
  default:
    throw std::runtime_error("Invalid op");
    break;
  }
      
  cv::Mat gxImageSigned, gyImageSigned;
  cv::filter2D(smoothImage, gxImageSigned, CV_16SC1, kernel.t(), anchor);
  cv::filter2D(smoothImage, gyImageSigned, CV_16SC1, kernel, anchor);
  const cv::Mat gxImage = cv::abs(gxImageSigned);
  const cv::Mat gyImage = cv::abs(gyImageSigned);
  if (sumFlag)
  {
    gradImage = gxImage + gyImage;
  }
  else
  {
    cv::sqrt(gxImage.mul(gxImage) + gyImage.mul(gyImage), gradImage);
  }
  gradImage.col(0).setTo(gradThresh - 1);
  gradImage.col(gradImage.cols - 1).setTo(gradThresh - 1);
  gradImage.row(0).setTo(gradThresh - 1);
  gradImage.row(gradImage.rows - 1).setTo(gradThresh - 1);
  gradImg = (short *)gradImage.data;

  dirImage = cv::Mat::zeros(gradImage.rows, gradImage.cols, CV_8UC1);
  const cv::Mat maskThresh = gradImage >= gradThresh;
  cv::Mat maskVertical, maskHorizontal;
  cv::bitwise_and(maskThresh, gxImage >= gyImage, maskVertical);
  cv::bitwise_and(maskThresh, gxImage < gyImage, maskHorizontal);
  dirImage.setTo(EDGE_VERTICAL, maskVertical);
  dirImage.setTo(EDGE_HORIZONTAL, maskHorizontal);
}

void ED::ComputeAnchorPoints()
{
  // memset(edgeImg, 0, width*height);
  for (int i = 2; i < height - 2; i++)
  {
    int start = 2;
    int inc = 1;
    if (i % scanInterval != 0)
    {
      start = scanInterval;
      inc = scanInterval;
    }

    for (int j = start; j < width - 2; j += inc)
    {
      if (gradImg[i * width + j] < gradThresh) continue;

      if (dirImg[i * width + j] == EDGE_VERTICAL)
      {
        // vertical edge
        int diff1 = gradImg[i * width + j] - gradImg[i * width + j - 1];
        int diff2 = gradImg[i * width + j] - gradImg[i * width + j + 1];
        if (diff1 >= anchorThresh && diff2 >= anchorThresh)
        {
          edgeImg[i * width + j] = ANCHOR_PIXEL;
          anchorPoints.push_back(Point(j, i));
        }
      }
      else
      {
        // horizontal edge
        int diff1 = gradImg[i * width + j] - gradImg[(i - 1) * width + j];
        int diff2 = gradImg[i * width + j] - gradImg[(i + 1) * width + j];
        if (diff1 >= anchorThresh && diff2 >= anchorThresh)
        {
          edgeImg[i * width + j] = ANCHOR_PIXEL;
          anchorPoints.push_back(Point(j, i));
        }
      }  // end-else
    }    // end-for-inner
  }      // end-for-outer

  anchorNos = anchorPoints.size();  // get the total number of anchor points
}

void ED::JoinAnchorPointsUsingSortedAnchors()
{
  std::vector<int> chainNos((width + height) * 8);
  std::vector<Point> pixels(width * height);
  std::vector<StackNode> stack(width * height);
  std::vector<Chain> chains(width * height);

  // sort the anchor points by their gradient value in decreasing order
  std::vector<int> A;
  sortAnchorsByGradValue1(A);

  // Now join the anchors starting with the anchor having the greatest gradient value
  int totalPixels = 0;

  for (int k = anchorNos - 1; k >= 0; k--)
  {
    int pixelOffset = A[k];

    int i = pixelOffset / width;
    int j = pixelOffset % width;

    // int i = anchorPoints[k].y;
    // int j = anchorPoints[k].x;

    if (edgeImg[i * width + j] != ANCHOR_PIXEL) continue;

    chains[0].len = 0;
    chains[0].parent = -1;
    chains[0].dir = 0;
    chains[0].children[0] = chains[0].children[1] = -1;
    chains[0].pixels = NULL;

    int noChains = 1;
    int len = 0;
    int duplicatePixelCount = 0;
    int top = -1;  // top of the stack

    if (dirImg[i * width + j] == EDGE_VERTICAL)
    {
      stack[++top].r = i;
      stack[top].c = j;
      stack[top].dir = DOWN;
      stack[top].parent = 0;

      stack[++top].r = i;
      stack[top].c = j;
      stack[top].dir = UP;
      stack[top].parent = 0;
    }
    else
    {
      stack[++top].r = i;
      stack[top].c = j;
      stack[top].dir = RIGHT;
      stack[top].parent = 0;

      stack[++top].r = i;
      stack[top].c = j;
      stack[top].dir = LEFT;
      stack[top].parent = 0;
    }  // end-else

    // While the stack is not empty
  StartOfWhile:
    while (top >= 0)
    {
      int r = stack[top].r;
      int c = stack[top].c;
      int dir = stack[top].dir;
      int parent = stack[top].parent;
      top--;

      if (edgeImg[r * width + c] != EDGE_PIXEL) duplicatePixelCount++;

      chains[noChains].dir = dir;  // traversal direction
      chains[noChains].parent = parent;
      chains[noChains].children[0] = chains[noChains].children[1] = -1;

      int chainLen = 0;

      chains[noChains].pixels = &pixels[len];

      pixels[len].y = r;
      pixels[len].x = c;
      len++;
      chainLen++;

      if (dir == LEFT)
      {
        while (dirImg[r * width + c] == EDGE_HORIZONTAL)
        {
          edgeImg[r * width + c] = EDGE_PIXEL;

          // The edge is horizontal. Look LEFT
          //
          //   A
          //   B x
          //   C
          //
          // cleanup up & down pixels
          if (edgeImg[(r - 1) * width + c] == ANCHOR_PIXEL) edgeImg[(r - 1) * width + c] = 0;
          if (edgeImg[(r + 1) * width + c] == ANCHOR_PIXEL) edgeImg[(r + 1) * width + c] = 0;

          // Look if there is an edge pixel in the neighbors
          if (edgeImg[r * width + c - 1] >= ANCHOR_PIXEL)
          {
            c--;
          }
          else if (edgeImg[(r - 1) * width + c - 1] >= ANCHOR_PIXEL)
          {
            r--;
            c--;
          }
          else if (edgeImg[(r + 1) * width + c - 1] >= ANCHOR_PIXEL)
          {
            r++;
            c--;
          }
          else
          {
            // else -- follow max. pixel to the LEFT
            int A = gradImg[(r - 1) * width + c - 1];
            int B = gradImg[r * width + c - 1];
            int C = gradImg[(r + 1) * width + c - 1];

            if (A > B)
            {
              if (A > C)
                r--;
              else
                r++;
            }
            else if (C > B)
              r++;
            c--;
          }  // end-else

          if (edgeImg[r * width + c] == EDGE_PIXEL || gradImg[r * width + c] < gradThresh)
          {
            if (chainLen > 0)
            {
              chains[noChains].len = chainLen;
              chains[parent].children[0] = noChains;
              noChains++;
            }  // end-if
            goto StartOfWhile;
          }  // end-else

          pixels[len].y = r;
          pixels[len].x = c;
          len++;
          chainLen++;
        }  // end-while

        stack[++top].r = r;
        stack[top].c = c;
        stack[top].dir = DOWN;
        stack[top].parent = noChains;

        stack[++top].r = r;
        stack[top].c = c;
        stack[top].dir = UP;
        stack[top].parent = noChains;

        len--;
        chainLen--;

        chains[noChains].len = chainLen;
        chains[parent].children[0] = noChains;
        noChains++;
      }
      else if (dir == RIGHT)
      {
        while (dirImg[r * width + c] == EDGE_HORIZONTAL)
        {
          edgeImg[r * width + c] = EDGE_PIXEL;

          // The edge is horizontal. Look RIGHT
          //
          //     A
          //   x B
          //     C
          //
          // cleanup up&down pixels
          if (edgeImg[(r + 1) * width + c] == ANCHOR_PIXEL) edgeImg[(r + 1) * width + c] = 0;
          if (edgeImg[(r - 1) * width + c] == ANCHOR_PIXEL) edgeImg[(r - 1) * width + c] = 0;

          // Look if there is an edge pixel in the neighbors
          if (edgeImg[r * width + c + 1] >= ANCHOR_PIXEL)
          {
            c++;
          }
          else if (edgeImg[(r + 1) * width + c + 1] >= ANCHOR_PIXEL)
          {
            r++;
            c++;
          }
          else if (edgeImg[(r - 1) * width + c + 1] >= ANCHOR_PIXEL)
          {
            r--;
            c++;
          }
          else
          {
            // else -- follow max. pixel to the RIGHT
            int A = gradImg[(r - 1) * width + c + 1];
            int B = gradImg[r * width + c + 1];
            int C = gradImg[(r + 1) * width + c + 1];

            if (A > B)
            {
              if (A > C)
                r--;  // A
              else
                r++;  // C
            }
            else if (C > B)
              r++;  // C
            c++;
          }  // end-else

          if (edgeImg[r * width + c] == EDGE_PIXEL || gradImg[r * width + c] < gradThresh)
          {
            if (chainLen > 0)
            {
              chains[noChains].len = chainLen;
              chains[parent].children[1] = noChains;
              noChains++;
            }  // end-if
            goto StartOfWhile;
          }  // end-else

          pixels[len].y = r;
          pixels[len].x = c;
          len++;
          chainLen++;
        }  // end-while

        stack[++top].r = r;
        stack[top].c = c;
        stack[top].dir = DOWN;  // Go down
        stack[top].parent = noChains;

        stack[++top].r = r;
        stack[top].c = c;
        stack[top].dir = UP;  // Go up
        stack[top].parent = noChains;

        len--;
        chainLen--;

        chains[noChains].len = chainLen;
        chains[parent].children[1] = noChains;
        noChains++;
      }
      else if (dir == UP)
      {
        while (dirImg[r * width + c] == EDGE_VERTICAL)
        {
          edgeImg[r * width + c] = EDGE_PIXEL;

          // The edge is vertical. Look UP
          //
          //   A B C
          //     x
          //
          // Cleanup left & right pixels
          if (edgeImg[r * width + c - 1] == ANCHOR_PIXEL) edgeImg[r * width + c - 1] = 0;
          if (edgeImg[r * width + c + 1] == ANCHOR_PIXEL) edgeImg[r * width + c + 1] = 0;

          // Look if there is an edge pixel in the neighbors
          if (edgeImg[(r - 1) * width + c] >= ANCHOR_PIXEL)
          {
            r--;
          }
          else if (edgeImg[(r - 1) * width + c - 1] >= ANCHOR_PIXEL)
          {
            r--;
            c--;
          }
          else if (edgeImg[(r - 1) * width + c + 1] >= ANCHOR_PIXEL)
          {
            r--;
            c++;
          }
          else
          {
            // else -- follow the max. pixel UP
            int A = gradImg[(r - 1) * width + c - 1];
            int B = gradImg[(r - 1) * width + c];
            int C = gradImg[(r - 1) * width + c + 1];

            if (A > B)
            {
              if (A > C)
                c--;
              else
                c++;
            }
            else if (C > B)
              c++;
            r--;
          }  // end-else

          if (edgeImg[r * width + c] == EDGE_PIXEL || gradImg[r * width + c] < gradThresh)
          {
            if (chainLen > 0)
            {
              chains[noChains].len = chainLen;
              chains[parent].children[0] = noChains;
              noChains++;
            }  // end-if
            goto StartOfWhile;
          }  // end-else

          pixels[len].y = r;
          pixels[len].x = c;

          len++;
          chainLen++;
        }  // end-while

        stack[++top].r = r;
        stack[top].c = c;
        stack[top].dir = RIGHT;
        stack[top].parent = noChains;

        stack[++top].r = r;
        stack[top].c = c;
        stack[top].dir = LEFT;
        stack[top].parent = noChains;

        len--;
        chainLen--;

        chains[noChains].len = chainLen;
        chains[parent].children[0] = noChains;
        noChains++;
      }
      else
      {  // dir == DOWN
        while (dirImg[r * width + c] == EDGE_VERTICAL)
        {
          edgeImg[r * width + c] = EDGE_PIXEL;

          // The edge is vertical
          //
          //     x
          //   A B C
          //
          // cleanup side pixels
          if (edgeImg[r * width + c + 1] == ANCHOR_PIXEL) edgeImg[r * width + c + 1] = 0;
          if (edgeImg[r * width + c - 1] == ANCHOR_PIXEL) edgeImg[r * width + c - 1] = 0;

          // Look if there is an edge pixel in the neighbors
          if (edgeImg[(r + 1) * width + c] >= ANCHOR_PIXEL)
          {
            r++;
          }
          else if (edgeImg[(r + 1) * width + c + 1] >= ANCHOR_PIXEL)
          {
            r++;
            c++;
          }
          else if (edgeImg[(r + 1) * width + c - 1] >= ANCHOR_PIXEL)
          {
            r++;
            c--;
          }
          else
          {
            // else -- follow the max. pixel DOWN
            int A = gradImg[(r + 1) * width + c - 1];
            int B = gradImg[(r + 1) * width + c];
            int C = gradImg[(r + 1) * width + c + 1];

            if (A > B)
            {
              if (A > C)
                c--;  // A
              else
                c++;  // C
            }
            else if (C > B)
              c++;  // C
            r++;
          }  // end-else

          if (edgeImg[r * width + c] == EDGE_PIXEL || gradImg[r * width + c] < gradThresh)
          {
            if (chainLen > 0)
            {
              chains[noChains].len = chainLen;
              chains[parent].children[1] = noChains;
              noChains++;
            }  // end-if
            goto StartOfWhile;
          }  // end-else

          pixels[len].y = r;
          pixels[len].x = c;

          len++;
          chainLen++;
        }  // end-while

        stack[++top].r = r;
        stack[top].c = c;
        stack[top].dir = RIGHT;
        stack[top].parent = noChains;

        stack[++top].r = r;
        stack[top].c = c;
        stack[top].dir = LEFT;
        stack[top].parent = noChains;

        len--;
        chainLen--;

        chains[noChains].len = chainLen;
        chains[parent].children[1] = noChains;
        noChains++;
      }  // end-else

    }  // end-while

    if (len - duplicatePixelCount < minPathLen)
    {
      for (int k = 0; k < len; k++)
      {
        edgeImg[pixels[k].y * width + pixels[k].x] = 0;
        edgeImg[pixels[k].y * width + pixels[k].x] = 0;

      }  // end-for
    }
    else
    {
      int noSegmentPixels = 0;

      int totalLen = LongestChain(chains, chains[0].children[1]);

      if (totalLen > 0)
      {
        // Retrieve the chainNos
        int count = RetrieveChainNos(chains, chains[0].children[1], chainNos);

        // Copy these pixels in the reverse order
        for (int k = count - 1; k >= 0; k--)
        {
          int chainNo = chainNos[k];

#if 1
          /* See if we can erase some pixels from the last chain. This is for cleanup */

          int fr = chains[chainNo].pixels[chains[chainNo].len - 1].y;
          int fc = chains[chainNo].pixels[chains[chainNo].len - 1].x;

          int index = noSegmentPixels - 2;
          while (index >= 0)
          {
            int dr = abs(fr - segmentPoints[segmentNos][index].y);
            int dc = abs(fc - segmentPoints[segmentNos][index].x);

            if (dr <= 1 && dc <= 1)
            {
              // neighbors. Erase last pixel
              segmentPoints[segmentNos].pop_back();
              noSegmentPixels--;
              index--;
            }
            else
              break;
          }  // end-while

          if (chains[chainNo].len > 1 && noSegmentPixels > 0)
          {
            fr = chains[chainNo].pixels[chains[chainNo].len - 2].y;
            fc = chains[chainNo].pixels[chains[chainNo].len - 2].x;

            int dr = abs(fr - segmentPoints[segmentNos][noSegmentPixels - 1].y);
            int dc = abs(fc - segmentPoints[segmentNos][noSegmentPixels - 1].x);

            if (dr <= 1 && dc <= 1) chains[chainNo].len--;
          }  // end-if
#endif

          for (int l = chains[chainNo].len - 1; l >= 0; l--)
          {
            segmentPoints[segmentNos].push_back(chains[chainNo].pixels[l]);
            noSegmentPixels++;
          }  // end-for

          chains[chainNo].len = 0;  // Mark as copied
        }                           // end-for
      }                             // end-if

      totalLen = LongestChain(chains, chains[0].children[0]);
      if (totalLen > 1)
      {
        // Retrieve the chainNos
        int count = RetrieveChainNos(chains, chains[0].children[0], chainNos);

        // Copy these chains in the forward direction. Skip the first pixel of the first chain
        // due to repetition with the last pixel of the previous chain
        int lastChainNo = chainNos[0];
        chains[lastChainNo].pixels++;
        chains[lastChainNo].len--;

        for (int k = 0; k < count; k++)
        {
          int chainNo = chainNos[k];

#if 1
          /* See if we can erase some pixels from the last chain. This is for cleanup */
          int fr = chains[chainNo].pixels[0].y;
          int fc = chains[chainNo].pixels[0].x;

          int index = noSegmentPixels - 2;
          while (index >= 0)
          {
            int dr = abs(fr - segmentPoints[segmentNos][index].y);
            int dc = abs(fc - segmentPoints[segmentNos][index].x);

            if (dr <= 1 && dc <= 1)
            {
              // neighbors. Erase last pixel
              segmentPoints[segmentNos].pop_back();
              noSegmentPixels--;
              index--;
            }
            else
              break;
          }  // end-while

          int startIndex = 0;
          int chainLen = chains[chainNo].len;
          if (chainLen > 1 && noSegmentPixels > 0)
          {
            int fr = chains[chainNo].pixels[1].y;
            int fc = chains[chainNo].pixels[1].x;

            int dr = abs(fr - segmentPoints[segmentNos][noSegmentPixels - 1].y);
            int dc = abs(fc - segmentPoints[segmentNos][noSegmentPixels - 1].x);

            if (dr <= 1 && dc <= 1)
            {
              startIndex = 1;
            }
          }  // end-if
#endif

          /* Start a new chain & copy pixels from the new chain */
          for (int l = startIndex; l < chains[chainNo].len; l++)
          {
            segmentPoints[segmentNos].push_back(chains[chainNo].pixels[l]);
            noSegmentPixels++;
          }  // end-for

          chains[chainNo].len = 0;  // Mark as copied
        }                           // end-for
      }                             // end-if

      // See if the first pixel can be cleaned up
      int fr = segmentPoints[segmentNos][1].y;
      int fc = segmentPoints[segmentNos][1].x;

      int dr = abs(fr - segmentPoints[segmentNos][noSegmentPixels - 1].y);
      int dc = abs(fc - segmentPoints[segmentNos][noSegmentPixels - 1].x);

      if (dr <= 1 && dc <= 1)
      {
        segmentPoints[segmentNos].erase(segmentPoints[segmentNos].begin());
        noSegmentPixels--;
      }  // end-if

      segmentNos++;
      segmentPoints.push_back(vector<Point>());  // create empty vector of points for segments

      // Copy the rest of the long chains here
      for (int k = 2; k < noChains; k++)
      {
        if (chains[k].len < 2) continue;

        totalLen = LongestChain(chains, k);

        if (totalLen >= 10)
        {
          // Retrieve the chainNos
          int count = RetrieveChainNos(chains, k, chainNos);

          // Copy the pixels
          noSegmentPixels = 0;
          for (int k = 0; k < count; k++)
          {
            int chainNo = chainNos[k];

#if 1
            /* See if we can erase some pixels from the last chain. This is for cleanup */
            int fr = chains[chainNo].pixels[0].y;
            int fc = chains[chainNo].pixels[0].x;

            int index = noSegmentPixels - 2;
            while (index >= 0)
            {
              int dr = abs(fr - segmentPoints[segmentNos][index].y);
              int dc = abs(fc - segmentPoints[segmentNos][index].x);

              if (dr <= 1 && dc <= 1)
              {
                // neighbors. Erase last pixel
                segmentPoints[segmentNos].pop_back();
                noSegmentPixels--;
                index--;
              }
              else
                break;
            }  // end-while

            int startIndex = 0;
            int chainLen = chains[chainNo].len;
            if (chainLen > 1 && noSegmentPixels > 0)
            {
              int fr = chains[chainNo].pixels[1].y;
              int fc = chains[chainNo].pixels[1].x;

              int dr = abs(fr - segmentPoints[segmentNos][noSegmentPixels - 1].y);
              int dc = abs(fc - segmentPoints[segmentNos][noSegmentPixels - 1].x);

              if (dr <= 1 && dc <= 1)
              {
                startIndex = 1;
              }
            }  // end-if
#endif
            /* Start a new chain & copy pixels from the new chain */
            for (int l = startIndex; l < chains[chainNo].len; l++)
            {
              segmentPoints[segmentNos].push_back(chains[chainNo].pixels[l]);
              noSegmentPixels++;
            }  // end-for

            chains[chainNo].len = 0;                 // Mark as copied
          }                                          // end-for
          segmentPoints.push_back(vector<Point>());  // create empty vector of points for segments
          segmentNos++;
        }  // end-if
      }    // end-for

    }  // end-else

  }  // end-for-outer

  // pop back last segment from vector
  // because of one preallocation in the beginning, it will always empty
  segmentPoints.pop_back();
}

void ED::sortAnchorsByGradValue()
{
  auto sortFunc = [&](const Point &a, const Point &b) {
    return gradImg[a.y * width + a.x] > gradImg[b.y * width + b.x];
  };

  std::sort(anchorPoints.begin(), anchorPoints.end(), sortFunc);

  /*
  ofstream myFile;
  myFile.open("anchorsNew.txt");
  for (int i = 0; i < anchorPoints.size(); i++) {
          int x = anchorPoints[i].x;
          int y = anchorPoints[i].y;

          myFile << i << ". value: " << gradImg[y*width + x] << "  Cord: (" << x << "," << y << ")"
  << endl;
  }
  myFile.close();


  vector<Point> temp(anchorNos);

  int x, y, i = 0;
  char c;
  std::ifstream infile("cords.txt");
  while (infile >> x >> c >> y && c == ',') {
          temp[i] = Point(x, y);
          i++;
  }

  anchorPoints = temp;
  */
}

void ED::sortAnchorsByGradValue1(std::vector<int> &A)
{
  const int SIZE = 128 * 256;
  std::vector<int> C(SIZE, 0);

  // Count the number of grad values
  for (int i = 1; i < height - 1; i++)
  {
    for (int j = 1; j < width - 1; j++)
    {
      if (edgeImg[i * width + j] != ANCHOR_PIXEL) continue;

      int grad = gradImg[i * width + j];
      C[grad]++;
    }  // end-for
  }    // end-for

  // Compute indices
  for (int i = 1; i < SIZE; i++) C[i] += C[i - 1];

  int noAnchors = C[SIZE - 1];
  A.clear();
  A.resize(noAnchors, 0);

  for (int i = 1; i < height - 1; i++)
  {
    for (int j = 1; j < width - 1; j++)
    {
      if (edgeImg[i * width + j] != ANCHOR_PIXEL) continue;

      int grad = gradImg[i * width + j];
      int index = --C[grad];
      A[index] = i * width + j;  // anchor's offset
    }                            // end-for
  }                              // end-for

  /*
  ofstream myFile;
  myFile.open("aNew.txt");
  for (int i = 0; i < noAnchors; i++)
          myFile << A[i] << endl;

  myFile.close(); */
}

int ED::LongestChain(std::vector<Chain> &chains, int root)
{
  if (root == -1 || chains[root].len == 0) return 0;

  int len0 = 0;
  if (chains[root].children[0] != -1) len0 = LongestChain(chains, chains[root].children[0]);

  int len1 = 0;
  if (chains[root].children[1] != -1) len1 = LongestChain(chains, chains[root].children[1]);

  int max = 0;

  if (len0 >= len1)
  {
    max = len0;
    chains[root].children[1] = -1;
  }
  else
  {
    max = len1;
    chains[root].children[0] = -1;
  }  // end-else

  return chains[root].len + max;
}  // end-LongestChain

int ED::RetrieveChainNos(std::vector<Chain> &chains, int root, std::vector<int> &chainNos)
{
  int count = 0;

  while (root != -1)
  {
    chainNos[count] = root;
    count++;

    if (chains[root].children[0] != -1)
      root = chains[root].children[0];
    else
      root = chains[root].children[1];
  }  // end-while

  return count;
}

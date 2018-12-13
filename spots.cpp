#include "DKImgPretreatment.h"
#include "DkImageStorage.h"
#include "DKImgHoleResults.h"
#include "DkSettings.h"
#include <QtMath>

namespace nmc {
	DKImgPretreatment::DKImgPretreatment(QImage orig,int mode)
	{
		mMode = mode;

		int cx = orig.width() >> 3;
		int cy = orig.width() >> 4;
		if (mMode == PRE_MODE::TIFF_MODE) {
			cx = orig.width() >> 6;
			cy = orig.width() >> 6;
		}
		double rgbMatArea = cx*cy;

		mResult = new DKImgHoleResults(rgbMatArea,cx,cy,mMode);
		mResult->setSrcPreview(orig);
		mResult->setGrayPreview(orig.convertToFormat(QImage::Format_RGB32));
	}

	DKImgHoleResults* DKImgPretreatment::getResult() {

		adaptiveThereshold();

		goToOpenCv();

		return mResult;
	}

	void DKImgPretreatment::covGrayImg()
	{
		QImage grayImg = mResult->getGrayPreview();

		for (int i = 0; i < grayImg.height(); i++) {
			for (int j = 0; j < grayImg.width(); j++) {
				QRgb grey = grayImg.pixel(j,i);

				int red = qRed(grey);
				int green = qGreen(grey);
				int blue = qBlue(grey);

				int g = (int)((float)red * 0.3 + (float)green * 0.59 + (float)blue * 0.11);
				grayImg.setPixel(j,i, qRgb(g, g, g));
			}
		}

		int p = 4;
		cv::Mat rgbMat = DkImage::qImage2Mat(grayImg);
		cv::Mat ele = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * p + 1, 2 * p + 1));
		cv::dilate(rgbMat, rgbMat, ele);
		mResult->setGrayPreview(DkImage::mat2QImage(rgbMat));
	}

	void DKImgPretreatment::getMinMaxGrayValue(int& minGrayValue, int& maxGrayValue) {
		QImage grayImg = mResult->getGrayPreview();

		double IMGPRE = 0.15;
		double IMGPRE_M = 1.0 - IMGPRE;

		for (int i = (int)(grayImg.height()*IMGPRE); i < (int)(grayImg.height()*IMGPRE_M); i++) {
			for (int j = (int)(grayImg.width()*IMGPRE); j <  (int)(grayImg.width()*IMGPRE_M); j++) {
				int gray = qBlue(grayImg.pixel(j, i));
				if (gray < minGrayValue)
					minGrayValue = gray;
				if (gray > maxGrayValue)
					maxGrayValue = gray;
			}
		}
	}

	void DKImgPretreatment::binarization(int T) {
		QImage grayImg = mResult->getGrayPreview();

		for (int i = 0; i < grayImg.height(); i++) {
			for (int j = 0; j < grayImg.width(); j++) {
				int gray = qBlue(grayImg.pixel(j, i));
				if (gray < T) {
					grayImg.setPixel(j, i, qRgb(0,0,0));
				}else {
					grayImg.setPixel(j, i, qRgb(255, 255, 255));
				}
			}
		}

		mResult->setBinPreview(grayImg);
	}

	int DKImgPretreatment::getIterationHresholdValue(int minGrayValue, int maxGrayValue) {
		QImage grayImg = mResult->getGrayPreview();

		double IMGPRE = 0.15;
		double IMGPRE_M = 1.0 - IMGPRE;

		int T1;
		int T2 = (maxGrayValue + minGrayValue) / 2;
		do {
			T1 = T2;
			float s = 0, l = 0;
			int    cs = 0, cl = 0;
			for (int i = (int)(grayImg.height()*IMGPRE); i < (int)(grayImg.height()*IMGPRE_M); i++) {
				for (int j = (int)(grayImg.width()*IMGPRE); j < (int)(grayImg.width()*IMGPRE_M); j++) {
					int gray = qBlue(grayImg.pixel(j, i));
					if (gray < T1) {
						s += gray;
						cs++;
					}
					if (gray > T1) {
						l += gray;
						cl++;
					}
				}
			}
			T2 = (int)(s / cs + l / cl) / 2;
		} while (T1 != T2);

		if (T1 <= (minGrayValue + 2) || T1 >= (maxGrayValue - 2))
			return (maxGrayValue + minGrayValue) / 2;

		return T1;
	}

	void DKImgPretreatment::goToOpenCv() {
		QImage binImg = mResult->getBinPreview();
		cv::Mat rgbMat = DkImage::qImage2Mat(binImg);

		int p = 3;
		if (mMode == PRE_MODE::TIFF_MODE) p = 2;
		cv::Mat ele = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * p, 2 * p));
		cv::dilate(rgbMat, rgbMat, ele);

		cv::Mat grayMat;
		cv::cvtColor(rgbMat, grayMat, cv::COLOR_RGB2GRAY);

		std::vector<std::vector<cv::Point>> contours;
		std::vector<cv::Vec4i> hierarchy;
		cv::findContours(grayMat, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

		filterContours(contours);
	}

	void DKImgPretreatment::filterContours(std::vector<std::vector<cv::Point>> contours) {

		QImage srcImg = mResult->getSrcPreview();
		cv::Mat rgbMat = DkImage::qImage2Mat(srcImg);

		for (int i = 0; i<contours.size(); i++)
		{
			cv::Moments mom = cv::moments(contours[i], true);
			//if (mom == null)
			//	continue;

			double area = mom.m00;
			{//面积过滤
				if (area < mResult->AREA_R_MIN || area > mResult->AREA_R_MAX) {
					continue;
				}
			}

			double arealen = 0.0f;
			{//周长比
				arealen = cv::arcLength(contours[i], true);
				double ratio = 4 * M_PI * area / (arealen * arealen);

				if (ratio <mResult->AREALEN_R_MIN || ratio > mResult->AREALEN_R_MAX) {
					continue;
				}
			}

			double r_len = 0.0f;
			{//半径
				r_len = 2 * area / arealen;
			}

			double denominator = 0.0f;
			/*{//惯性系
				denominator = qSqrt(qPow(2 * mom.mu11, 2) + qPow(mom.mu20 - mom.mu02, 2));
				double eps = 1e-2;
				double ratio;
				if (denominator > eps)
				{
					double cosmin = (mom.mu20 - mom.mu02) / denominator;
					double sinmin = 2 * mom.mu11 / denominator;
					double cosmax = -cosmin;
					double sinmax = -sinmin;

					double imin = 0.5 * (mom.mu20 + mom.mu02) - 0.5 * (mom.mu20 - mom.mu02) * cosmin - mom.mu11 * sinmin;
					double imax = 0.5 * (mom.mu20 + mom.mu02) - 0.5 * (mom.mu20 - mom.mu02) * cosmax - mom.mu11 * sinmax;
					ratio = imin / imax;
				}
				else
				{
					ratio = 1;
				}

				if (ratio < mResult->DENOMINATOR_R_MIN || ratio > mResult->DENOMINATOR_R_MAX) {
					continue;
				}
			}*/

			double hullarea = 0.0f;
			std::vector<cv::Point> hullPoints;
			{//凸包面积
				std::vector<int> hull;
				cv::convexHull(contours[i], hull);

				for (int n = 0; n < hull.size(); n++)
				{
					hullPoints.push_back(contours[i][hull[n]]);
				}

				hullarea = cv::contourArea(hullPoints);
				double a = cv::contourArea(contours[i]);
				double ratio = a / hullarea;

				if (ratio < mResult->HULLAREA_R_MIN || ratio > mResult->HULLAREA_R_MAX) {
					continue;
				}
			}

			QPoint center;
			{//质心
				int x = (int)(mom.m10 / mom.m00);
				int y = (int)(mom.m01 / mom.m00);

				center.setX(x);
				center.setY(y);

				if (x<0 || y<0 || x >= srcImg.width() || y >= srcImg.height()) {
					continue;
				}
			}

			{//获取结果
				DKImgHoleResult* c = new DKImgHoleResult();
				c->setArea(area);
				c->setAreaLen(arealen);
				c->setRad(r_len);
				c->setCenter(center);
				c->setDenominator(denominator);
				c->setHullArea(hullarea);
				c->setContour(hullPoints);
				c->setRepColor(getXYColor(center.x(), center.y(), (int)(r_len*0.5)));

				c->setColor(srcImg.pixel(center.x(), center.y()));

				bool binsert = false;
				QVector<DKImgHoleResult*> results = mResult->getContoursResults();
				for (int n = 0; n < results.size(); n++)
				{
					QPoint nPoint = results[n]->getCenter();
					double nRad = results[n]->getRad();

					double rad = nRad > r_len ? nRad : r_len;
					if ((nPoint.y() - center.y()) > rad) {
						binsert = true;
						mResult->insertAt(c, n);
						break;
					}
					else {
						if (center.x() <= nPoint.x()) {
							binsert = true;
							mResult->insertAt(c, n);
							break;
						}
					}
				}

				if (!binsert) {
					mResult->add(c);
				}

			}
		}

		QVector<DKImgHoleResult*> results = mResult->getContoursResults();
		for (int i = 0; i < results.size(); i++)
		{
			DKImgHoleResult* pResult = results[i];
			cv::Scalar color(rand() & 255, rand() & 255, rand() & 255, 255);

			std::vector<std::vector<cv::Point>> drawContours;
			drawContours.push_back(pResult->getContour());
			cv::drawContours(rgbMat, drawContours, 0, color, 1, 4);

			double v = QString::asprintf("%d", i + 1).length()*0.5;
			cv::putText(rgbMat, QString::asprintf("%d", i + 1).toStdString(), cv::Point(pResult->getCenter().x() - pResult->getRad()*v, pResult->getCenter().y() + pResult->getRad()*0.5), CV_FONT_HERSHEY_PLAIN, pResult->getRad() / 10, color, 1, 1);
		}

		mResult->setResultPreview(DkImage::mat2QImage(rgbMat));
	}

	int DKImgPretreatment::getXYColor(int x, int y, int r0)
	{
		QImage srcImg = mResult->getSrcPreview();

		r0 = (r0 <= 2) ? 2 : r0;

		int startx = x - r0;
		int starty = y - r0;
		int endx = x + r0;
		int endy = y + r0;

		if (startx<0 || starty<0 || endx >= srcImg.width() || endy >= srcImg.height()) {
			int r = qRed(srcImg.pixel(x, y));
			int g = qGreen(srcImg.pixel(x, y));
			int b = qBlue(srcImg.pixel(x, y));

			return (int)((float)r * 0.3 + (float)g * 0.59 + (float)b * 0.11);
		}

		int greys[256] = {0};
		int num = 0;
		for (int i = starty; i <= endy; i++)
		{
			for (int j = startx; j <= endx; j++)
			{
				int r = qRed(srcImg.pixel(j, i));
				int g = qGreen(srcImg.pixel(j, i));
				int b = qBlue(srcImg.pixel(j, i));

				int grey = (int)((float)r * 0.3 + (float)g * 0.59 + (float)b * 0.11);

				greys[grey]++;
				num++;
			}
		}

		QVector<float> grey_ps;
		for (int i = 0; i<256; i++) {
			float p = ((float)greys[i]) / ((float)num);
			if (p <= 0.0f)
				continue;

			bool bInsert = false;
			for (int j = 0; j<grey_ps.size(); j++) {
				if (p >grey_ps[j]) {
					grey_ps.insert(j, p);
					bInsert = true;
					break;
				}
			}

			if (!bInsert) {
				grey_ps.push_back(p);
			}
		}

		float P_GREY = 0.0001f;
		if (grey_ps.size() >= 5) {
			P_GREY = grey_ps[(grey_ps.size() / 2 - 1)];
		}

		int greynum = 0;
		for (int i = 0; i<256; i++) {
			float p = ((float)greys[i]) / ((float)num);
			if (p < P_GREY)
				continue;

			greynum += greys[i];
		}

		float grey = 0;
		for (int i = 0; i<256; i++) {
			float p = ((float)greys[i]) / ((float)num);
			if (p < P_GREY)
				continue;

			grey += i*((float)greys[i] / (float)greynum);
		}

		return (int)grey;
	}

	int DKImgPretreatment::getBinarizationT()
	{
		QImage grayImg = mResult->getGrayPreview();

		double S_IMGPRE = 0.15;
		double E_IMGPRE = 0.30;
		double E_IMGPRE_M = 1.0 - E_IMGPRE;
		double S_IMGPRE_M = 1.0 - S_IMGPRE;

		int grays[256] = {0};
		int num = 0;
		double allgray = 0.0;

		for (int i = (int)(grayImg.width()*S_IMGPRE); i < (int)(grayImg.width()*E_IMGPRE); i++) {
			for (int j = (int)(grayImg.height()*S_IMGPRE); j < (int)(grayImg.height()*S_IMGPRE_M); j++) {
				int gray = qBlue(grayImg.pixel(i, j));
				grays[gray]++;
				num++;
				allgray += gray;
			}
		}

		for (int i = (int)(grayImg.width()*E_IMGPRE_M); i < (int)(grayImg.width()*S_IMGPRE_M); i++) {
			for (int j = (int)(grayImg.height()*S_IMGPRE); j < (int)(grayImg.height()*S_IMGPRE_M); j++) {
				int gray = qBlue(grayImg.pixel(i, j));
				grays[gray]++;
				num++;
				allgray += gray;
			}
		}

		for (int i = (int)(grayImg.width()*E_IMGPRE); i < (int)(grayImg.width()*E_IMGPRE_M); i++) {
			for (int j = (int)(grayImg.height()*S_IMGPRE); j < (int)(grayImg.height()*E_IMGPRE); j++) {
				int gray = qBlue(grayImg.pixel(i, j));
				grays[gray]++;
				num++;
				allgray += gray;
			}
		}

		for (int i = (int)(grayImg.width()*E_IMGPRE); i < (int)(grayImg.width()*E_IMGPRE_M); i++) {
			for (int j = (int)(grayImg.height()*E_IMGPRE_M); j < (int)(grayImg.height()*S_IMGPRE_M); j++) {
				int gray = qBlue(grayImg.pixel(i, j));
				grays[gray]++;
				num++;
				allgray += gray;
			}
		}

		double gray = allgray / num;

		int numValue = 0;
		for (int i = 0; i < 256; i++)
		{
			if (grays[i] > 10) numValue++;
		}

		gray = gray - ((double)numValue*0.5*0.60);

		return gray;
	}

	void DKImgPretreatment::adaptiveThereshold()
	{
		QImage grayImg = mResult->getGrayPreview();
		cv::Mat src = DkImage::qImage2Mat(grayImg);

		cv::Mat dst;
		cv::cvtColor(src, dst, cv::COLOR_RGB2GRAY);
		int x1, y1, x2, y2;
		int count = 0;
		long long sum = 0;

		int S = (double)(src.rows >> 3) - (double)(src.rows >> 3)*0.1;  //划分区域的大小S*S  
		int T = mResult->IMGPRETREATMENT; /*百分比，用来最后与阈值的比较。原文：If the value of the current pixel is t percent less than this average then it is set to black, otherwise it is set to white.*/
		if (mMode == PRE_MODE::TIFF_MODE) {
			S = src.rows >> 6;
			T = mResult->IMGPRETREATMENT;
		}

		int W = dst.cols;
		int H = dst.rows;
		long long **Argv;
		Argv = new long long*[dst.rows];
		for (int ii = 0; ii<dst.rows; ii++)
		{
			Argv[ii] = new long long[dst.cols];
		}

		for (int i = 0; i<W; i++)
		{
			sum = 0;
			for (int j = 0; j<H; j++)
			{
				sum += dst.at<uchar>(j, i);
				if (i == 0)
					Argv[j][i] = sum;
				else
					Argv[j][i] = Argv[j][i - 1] + sum;
			}
		}

		for (int i = 0; i<W; i++)
		{
			for (int j = 0; j<H; j++)
			{
				x1 = i - S / 2;
				x2 = i + S / 2;
				y1 = j - S / 2;
				y2 = j + S / 2;
				if (x1<0)
					x1 = 0;
				if (x2 >= W)
					x2 = W - 1;
				if (y1<0)
					y1 = 0;
				if (y2 >= H)
					y2 = H - 1;
				count = (x2 - x1)*(y2 - y1);
				sum = Argv[y2][x2] - Argv[y1][x2] - Argv[y2][x1] + Argv[y1][x1];


				if ((long long)(dst.at<uchar>(j, i)*count)<(long long)sum*(100 - T) / 100)
					dst.at<uchar>(j, i) = 0;
				else
					dst.at<uchar>(j, i) = 255;
			}
		}
		for (int i = 0; i < dst.rows; ++i)
		{
			delete[] Argv[i];
		}
		delete[] Argv;

		cv::cvtColor(dst, dst, cv::COLOR_GRAY2RGB);
		mResult->setBinPreview(DkImage::mat2QImage(dst));
	}

}

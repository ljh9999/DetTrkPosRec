#include "tracker.h"

xsk::Point::Point(int x_, int y_)
{
    x = x_;
    y = y_;
}

xsk::Rect::Rect()
{
    x = 0;
    y = 0;
    width = 0;
    height = 0;
}

xsk::Rect::Rect(int x_, int y_, int width_, int height_)
{
    x = x_;
    y = y_;
    width = width_;
    height = height_;
}

xsk::Point xsk::Rect::tl() const
{
    return {x, y};
}

xsk::Point xsk::Rect::br() const
{
    return {x + width, y + height};
}

int xsk::Rect::area() const
{
    return width * height;
}


//xsk::Rect &xsk::Rect::operator=(xsk::Rect &obj)
//{
//    x = obj.x;
//    y = obj.y;
//    width = obj.width;
//    height = obj.height;
//    return *this;
//}


Tracker::Tracker() : kf_(8, 4)
{

    /*** Define constant velocity model ***/
    // state - center_x, center_y, width, height, v_cx, v_cy, v_width, v_height
    kf_.F_ <<
           1, 0, 0, 0, 1, 0, 0, 0,
            0, 1, 0, 0, 0, 1, 0, 0,
            0, 0, 1, 0, 0, 0, 1, 0,
            0, 0, 0, 1, 0, 0, 0, 1,
            0, 0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 1;

    // Give high uncertainty to the unobservable initial velocities
    kf_.P_ <<
           10, 0, 0, 0, 0, 0, 0, 0,
            0, 10, 0, 0, 0, 0, 0, 0,
            0, 0, 10, 0, 0, 0, 0, 0,
            0, 0, 0, 10, 0, 0, 0, 0,
            0, 0, 0, 0, 10000, 0, 0, 0,
            0, 0, 0, 0, 0, 10000, 0, 0,
            0, 0, 0, 0, 0, 0, 10000, 0,
            0, 0, 0, 0, 0, 0, 0, 10000;


    kf_.H_ <<
           1, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0;

    kf_.Q_ <<
           1, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0.01, 0, 0, 0,
            0, 0, 0, 0, 0, 0.01, 0, 0,
            0, 0, 0, 0, 0, 0, 0.0001, 0,
            0, 0, 0, 0, 0, 0, 0, 0.0001;

    kf_.R_ <<
           1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 10, 0,
            0, 0, 0, 10;
}


// Get predicted locations from existing trackers
// dt is time elapsed between the current and previous measurements
void Tracker::Predict()
{
    kf_.Predict();

    // hit streak count will be reset
    if (coast_cycles_ > 0)
    {
        hit_streak_ = 0;
    }
    // accumulate coast cycle count
    coast_cycles_++;
}


// Update matched trackers with assigned detections
//void Tracker::Update(const cv::Rect &bbox)
//{
//    // get measurement update, reset coast cycle count
//    coast_cycles_ = 0;
//    // accumulate hit streak count
//    hit_streak_++;
//
//    // observation - center_x, center_y, area, ratio
//    Eigen::VectorXd observation = ConvertBboxToObservation(bbox);
//    kf_.Update(observation);
//}
void Tracker::Update(const xsk::Rect &bbox)
{
    // get measurement update, reset coast cycle count
    coast_cycles_ = 0;
    // accumulate hit streak count
    hit_streak_++;

    // observation - center_x, center_y, area, ratio
    Eigen::VectorXd observation = ConvertBboxToObservation(bbox);
    kf_.Update(observation);
}


// Create and initialize new trackers for unmatched detections, with initial bounding box
//void Tracker::Init(const cv::Rect &bbox)
//{
//    kf_.x_.head(4) << ConvertBboxToObservation(bbox);
//    hit_streak_++;
//}

void Tracker::Init(const xsk::Rect &bbox)
{
    kf_.x_.head(4) << ConvertBboxToObservation(bbox);
    hit_streak_++;
}


/**
 * Returns the current bounding box estimate
 * @return
 */
//cv::Rect Tracker::GetStateAsBbox() const
//{
//    return ConvertStateToBbox(kf_.x_);
//}
xsk::Rect Tracker::GetStateAsBbox() const
{
    return ConvertStateToBbox(kf_.x_);
}


float Tracker::GetNIS() const
{
    return kf_.NIS_;
}


/**
 * Takes a bounding box in the form [x, y, width, height] and returns z in the form
 * [x, y, s, r] where x,y is the centre of the box and s is the scale/area and r is
 * the aspect ratio
 *
 * @param bbox
 * @return
 */
//Eigen::VectorXd Tracker::ConvertBboxToObservation(const cv::Rect &bbox) const
//{
//    Eigen::VectorXd observation = Eigen::VectorXd::Zero(4);
//    auto width = static_cast<float>(bbox.width);
//    auto height = static_cast<float>(bbox.height);
//    float center_x = bbox.x + width / 2;
//    float center_y = bbox.y + height / 2;
//    observation << center_x, center_y, width, height;
//    return observation;
//}
Eigen::VectorXd Tracker::ConvertBboxToObservation(const xsk::Rect &bbox) const
{
    Eigen::VectorXd observation = Eigen::VectorXd::Zero(4);
    auto width = static_cast<float>(bbox.width);
    auto height = static_cast<float>(bbox.height);
    float center_x = float(bbox.x) + width / 2;
    float center_y = float(bbox.y) + height / 2;
    observation << center_x, center_y, width, height;
    return observation;
}

/**
 * Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
 * [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
 *
 * @param state
 * @return
 */
//cv::Rect Tracker::ConvertStateToBbox(const Eigen::VectorXd &state) const
//{
//    // state - center_x, center_y, width, height, v_cx, v_cy, v_width, v_height
//    auto width = static_cast<int>(state[2]);
//    auto height = static_cast<int>(state[3]);
//    auto tl_x = static_cast<int>(state[0] - width / 2.0);
//    auto tl_y = static_cast<int>(state[1] - height / 2.0);
//    cv::Rect rect(cv::Point(tl_x, tl_y), cv::Size(width, height));
//    return rect;
//}
xsk::Rect Tracker::ConvertStateToBbox(const Eigen::VectorXd &state) const
{
    // state - center_x, center_y, width, height, v_cx, v_cy, v_width, v_height
    auto width = static_cast<int>(state[2]);
    auto height = static_cast<int>(state[3]);
    auto tl_x = static_cast<int>(state[0] - width / 2.0);
    auto tl_y = static_cast<int>(state[1] - height / 2.0);
    xsk::Rect rect{tl_x, tl_y, width, height};
    return rect;
}

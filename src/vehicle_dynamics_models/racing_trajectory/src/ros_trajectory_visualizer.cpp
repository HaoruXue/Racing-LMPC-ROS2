// Copyright 2023 Haoru Xue
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include <memory>
#include <lmpc_utils/ros_param_helper.hpp>
#include "racing_trajectory/ros_trajectory_visualizer.hpp"

namespace lmpc
{
namespace vehicle_model
{
namespace racing_trajectory
{

ROSTrajectoryVisualizer::ROSTrajectoryVisualizer(RacingTrajectory & trajectory)
{
  change_trajectory(trajectory);
}

void ROSTrajectoryVisualizer::change_trajectory(RacingTrajectory & trajectory)
{
  std::unique_lock<std::shared_mutex> lock(mutex_);
  const auto abscissa =
    casadi::DM::linspace(0.0, trajectory.total_length(), 1001)(casadi::Slice(0, -1));
  const auto N = abscissa.size1();

  abscissa_polygon_msg_ = std::make_shared<PolygonStamped>();
  abscissa_polygon_msg_->header.frame_id = "map";
  const auto pts = casadi::DM::horzcat(
        {
          trajectory.x_interpolation_function()(abscissa)[0],
          trajectory.y_interpolation_function()(abscissa)[0],
          casadi::DM::zeros(N)
        }).T();
  abscissa_polygon_msg_->polygon = build_polygon(pts);

  left_boundary_polygon_msg_ = std::make_shared<PolygonStamped>();
  left_boundary_polygon_msg_->header.frame_id = "map";
  right_boundary_polygon_msg_ = std::make_shared<PolygonStamped>();
  right_boundary_polygon_msg_->header.frame_id = "map";
  surface_marker_msg_ = std::make_shared<MarkerArray>();
  const auto left_boundary_frenet = casadi::DM::horzcat(
        {
          abscissa,
          trajectory.left_boundary_interpolation_function()(abscissa)[0],
          casadi::DM::zeros(N)
        }).T();
  const auto right_boundary_frenet = casadi::DM::horzcat(
        {
          abscissa,
          trajectory.right_boundary_interpolation_function()(abscissa)[0],
          casadi::DM::zeros(N)
        }).T();
  const auto bank_angles = trajectory.bank_interpolation_function()(abscissa)[0].T();
  const auto left_boundary_elevation = left_boundary_frenet(1, casadi::Slice()) * casadi::DM::sin(
    bank_angles);
  const auto right_boundary_elevation = right_boundary_frenet(1, casadi::Slice()) * casadi::DM::sin(
    bank_angles);
  const auto left_boundary_global =
    trajectory.frenet_to_global_function().map(N)(left_boundary_frenet)[0];
  const auto right_boundary_global =
    trajectory.frenet_to_global_function().map(N)(right_boundary_frenet)[0];
  const auto left_polygon_data = casadi::DM::vertcat(
        {
          left_boundary_global(casadi::Slice(0, 2), casadi::Slice()),
          left_boundary_elevation
        });
  const auto right_polygon_data = casadi::DM::vertcat(
        {
          right_boundary_global(casadi::Slice(0, 2), casadi::Slice()),
          right_boundary_elevation
        });
  left_boundary_polygon_msg_->polygon = build_polygon(left_polygon_data);
  right_boundary_polygon_msg_->polygon = build_polygon(right_polygon_data);
  surface_marker_msg_->markers.push_back(build_surface(left_polygon_data, right_polygon_data));
  surface_marker_msg_->markers.back().header.frame_id = "map";
}

ROSTrajectoryVisualizer::~ROSTrajectoryVisualizer()
{
  if (static_vis_timer_) {
    static_vis_timer_->cancel();
    static_vis_timer_.reset();
  }

  if (left_boundary_polygon_pub_) {
    left_boundary_polygon_pub_.reset();
  }

  if (right_boundary_polygon_pub_) {
    right_boundary_polygon_pub_.reset();
  }

  if (abscissa_polygon_pub_) {
    abscissa_polygon_pub_.reset();
  }
}

void ROSTrajectoryVisualizer::attach_ros_publishers(
  rclcpp::Node * node, const double & dt,
  const bool & vis_boundary,
  const bool & vis_abscissa)
{
  node_ = node;
  visualize_3d_ = lmpc::utils::declare_parameter<bool>(node_, "visualize_3d", true);
  if (vis_boundary) {
    left_boundary_polygon_pub_ = node->create_publisher<PolygonStamped>(
      "left_boundary_polygon", 1);
    right_boundary_polygon_pub_ = node->create_publisher<PolygonStamped>(
      "right_boundary_polygon", 1);
    marker_pub_ = node->create_publisher<visualization_msgs::msg::MarkerArray>(
      "track_surface", 1);
  }
  if (vis_abscissa) {
    abscissa_polygon_pub_ = node->create_publisher<PolygonStamped>(
      "abscissa_polygon", 1);
  }

  vis_callback_group_ = node->create_callback_group(
    rclcpp::CallbackGroupType::MutuallyExclusive);
  static_vis_timer_ = node->create_wall_timer(
    std::chrono::milliseconds(static_cast<int>(dt * 1000.0)),
    std::bind(&ROSTrajectoryVisualizer::on_static_vis_timer, this), vis_callback_group_);
}

Polygon ROSTrajectoryVisualizer::build_polygon(const casadi::DM & pts)
{
  Polygon polygon;
  polygon.points.reserve(pts.size2());
  for (int i = 0; i < pts.size2(); ++i) {
    auto & pt = polygon.points.emplace_back();
    pt.x = static_cast<double>(pts(0, i));
    pt.y = static_cast<double>(pts(1, i));
    if (visualize_3d_) {
      pt.z = static_cast<double>(pts(2, i));
    }
  }
  return polygon;
}

Marker ROSTrajectoryVisualizer::build_surface(const casadi::DM & left, const casadi::DM & right)
{
  if (left.size2() != right.size2()) {
    throw std::runtime_error("left and right must have the same number of columns");
  }
  Marker marker;
  marker.type = Marker::TRIANGLE_LIST;
  marker.points.reserve(left.size2() * 6);
  marker.id = 0;
  marker.ns = "track_surface";
  marker.scale.x = 1.0;
  marker.scale.y = 1.0;
  marker.scale.z = 1.0;

  auto add_pt = [&marker](const casadi_int & i, const casadi::DM & pt) {
      auto & pt1 = marker.points.emplace_back();
      pt1.x = static_cast<double>(pt(0, i));
      pt1.y = static_cast<double>(pt(1, i));
      pt1.z = static_cast<double>(pt(2, i));
    };

  for (casadi_int i = 0; i < left.size2(); ++i) {
    const casadi_int i_next = (i + 1) % left.size2();
    add_pt(i, left);
    add_pt(i, right);
    add_pt(i_next, left);
    add_pt(i_next, left);
    add_pt(i, right);
    add_pt(i_next, right);
  }
  return marker;
}

void ROSTrajectoryVisualizer::on_static_vis_timer()
{
  std::shared_lock<std::shared_mutex> lock(mutex_);
  const auto now = node_->now();
  if (abscissa_polygon_pub_) {
    abscissa_polygon_msg_->header.stamp = now;
    abscissa_polygon_pub_->publish(*abscissa_polygon_msg_);
  }
  if (left_boundary_polygon_pub_ && right_boundary_polygon_pub_ && marker_pub_) {
    left_boundary_polygon_msg_->header.stamp = now;
    left_boundary_polygon_pub_->publish(*left_boundary_polygon_msg_);
    right_boundary_polygon_msg_->header.stamp = now;
    right_boundary_polygon_pub_->publish(*right_boundary_polygon_msg_);
    surface_marker_msg_->markers.front().header.stamp = now;
    marker_pub_->publish(*surface_marker_msg_);
  }
}
}  // namespace racing_trajectory
}  // namespace vehicle_model
}  // namespace lmpc

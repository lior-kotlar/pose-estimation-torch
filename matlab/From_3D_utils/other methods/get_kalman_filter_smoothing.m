function kalman_smoothed_pts = get_kalman_filter_smoothing(points_3D, std_matrix)
%UNTITLED11 Summary of this function goes here
%   Detailed explanation goes here
% Set up the Kalman filter
num_joints = size(points_3D, 1);
num_frames = size(points_3D, 2);
mean_std = mean(std_matrix) * 100;
kalman_smoothed_pts = zeros(size(points_3D));
for joint=1:num_joints
    points = squeeze(points_3D(joint, :, :));
    n = 6; % State size
    m = 3; % Measurement size
    F = [1 0 0 1 0 0; 0 1 0 0 1 0; 0 0 1 0 0 1; 
        0 0 0 1 0 0; 0 0 0 0 1 0; 0 0 0 0 0 1]; % State transition matrix
    H = [1 0 0 0 0 0; 0 1 0 0 0 0; 0 0 1 0 0 0]; % Measurement matrix
    Q = mean_std*eye(n); % Process noise covariance
    R = mean_std*eye(m); % Measurement noise covariance
    x_est = zeros(n, 1); % Initial state estimate
    P_est = eye(n); % Initial state covariance estimate
    
    % Apply the Kalman filter
    for i = 1:num_frames
       % Predict the next state
       x_pred = F*x_est;
       P_pred = F*P_est*F' + Q;
       
       % Update the state estimate
       K = P_pred*H' / (H*P_pred*H' + R); % Kalman gain
       y = points(i,:)' - H*x_pred; % Measurement residual
       x_est = x_pred + K*y;
       P_est = (eye(n) - K*H)*P_pred;
       
       % Store the filtered estimates
       x_filtered(i,:) = x_est';
    end
    kalman_smoothed_pts(joint, :, :) = x_filtered(:, 1:3);

%     plot3(points(:,1), points(:,2), points(:,3), 'b-');
%     hold on;
%     plot3(x_filtered(:,1), x_filtered(:,2), x_filtered(:,3), 'r--');
end
% Plot the original points and the filtered points

end
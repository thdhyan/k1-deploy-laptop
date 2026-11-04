#!/usr/bin/env python3
"""Generate RViz config for 6-robot fleet (18 camera streams + TF + RobotModel)."""
import os
import sys

ROBOTS = ['k1_0', 'k1_1', 'k1_2', 'k1_3', 'k1_4', 'k1_5']
ALIASES = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3']

CAMERA_TOPICS = [
    ('/boostercamera/head/rgb',       'Left RGB'),    # from compressed stream
    ('/boostercamera/head/raw/right/rgb', 'Right RGB'),
    ('/boostercamera/head/depth',         'Depth'),
]

def make_image_display(ns, alias, cam_topic, cam_label, panel_x, panel_y, w, h):
    """Generate an rviz Image display YAML block."""
    display_name = f'{alias} {cam_label}'
    topic = f'/{ns}{cam_topic}'
    return f"""
      - Class: rviz_default_plugins/Image
        Enabled: true
        Name: {display_name}
        Topic:
          Value: {topic}
          Depth: 5
          Durability Policy: Volatile
          History Policy: Keep Last
          Reliability Policy: Best Effort
        Value: true
        Image Rendering Panel:
          Panel X: {panel_x}
          Panel Y: {panel_y}
          Panel Width: {w}
          Panel Height: {h}
"""


def make_tf_display():
    return """
      - Class: rviz_default_plugins/TF
        Enabled: true
        Name: TF
        Show Arrows: true
        Show Axes: true
        Show Names: true
        Frame Timeout: 15
        Frames:
          All Enabled: true
        Value: true
"""


def make_robot_model_display(ns, alias):
    return f"""
      - Class: rviz_default_plugins/RobotModel
        Enabled: true
        Name: {alias} Robot
        Robot Description: robot_description
        Visual Enabled: true
        Collision Enabled: false
        Alpha: 1.0
        Axes Prefix: {ns}/
        Name: {alias}
        Value: true
"""


def make_3d_panel():
    return """
      - Class: rviz_default_plugins/TF
        Enabled: true
        Name: TF (3D)
        Show Arrows: true
        Show Axes: true
        Show Names: true
        Value: true
"""


def generate_config():
    """Build the complete RViz config YAML."""
    displays = []

    # 18 image displays in a 3x6 grid
    COLS = 3
    for robot_idx, (ns, alias) in enumerate(zip(ROBOTS, ALIASES)):
        for cam_idx, (cam_topic, cam_label) in enumerate(CAMERA_TOPICS):
            col = cam_idx
            row = robot_idx
            display_name = f'{alias} {cam_label}'
            topic = f'/{ns}{cam_topic}'
            displays.append({
                'Class': 'rviz_default_plugins/Image',
                'Enabled': True,
                'Name': display_name,
                'Topic': {
                    'Value': topic,
                    'Depth': 5,
                    'Durability Policy': 'Volatile',
                    'History Policy': 'Keep Last',
                    'Reliability Policy': 'Best Effort',
                },
                'Value': True,
            })

    # TF display
    displays.append({
        'Class': 'rviz_default_plugins/TF',
        'Enabled': True,
        'Name': 'TF',
        'Show Arrows': True,
        'Show Axes': True,
        'Show Names': True,
        'Frame Timeout': 15,
        'Value': True,
    })

    # RobotModel per robot
    for ns, alias in zip(ROBOTS, ALIASES):
        displays.append({
            'Class': 'rviz_default_plugins/RobotModel',
            'Enabled': True,
            'Name': f'{alias} Robot',
            'Description Source': 'Topic',
            'Robot Description Topic': {
                'Value': f'/{ns}/robot_description',
                'Depth': 5,
                'Durability Policy': 'Transient Local',
                'History Policy': 'Keep Last',
                'Reliability Policy': 'Reliable',
            },
            'Visual Enabled': True,
            'Collision Enabled': False,
            'Alpha': 1.0,
            'Value': True,
        })

    # Build YAML manually (more reliable than ruamel.yaml)
    lines = []
    lines.append('Panels:')
    lines.append('  - Class: rviz_common/Displays')
    lines.append('    Name: Displays')
    lines.append('    Property Tree Widget:')
    lines.append('      Expand Groups: true')
    lines.append('      Name: Displays')
    lines.append('    Value: true')
    lines.append('  - Class: rviz_common/Views')
    lines.append('    Name: Views')
    lines.append('    Value: true')
    lines.append('')
    lines.append('Visualization Manager:')
    lines.append('  Displays:')

    # Image displays (grouped by robot)
    for robot_idx, (ns, alias) in enumerate(zip(ROBOTS, ALIASES)):
        lines.append(f'    - Class: rviz_common/Group')
        lines.append(f'      Name: {alias} ({ns})')
        lines.append(f'      Value: true')
        for cam_idx, (cam_topic, cam_label) in enumerate(CAMERA_TOPICS):
            topic = f'/{ns}{cam_topic}'
            display_name = f'{cam_label}'
            lines.append(f'        - Class: rviz_default_plugins/Image')
            lines.append(f'          Enabled: true')
            lines.append(f'          Name: {display_name}')
            lines.append(f'          Topic:')
            lines.append(f'            Value: {topic}')
            lines.append(f'            Depth: 5')
            lines.append(f'            Durability Policy: Volatile')
            lines.append(f'            History Policy: Keep Last')
            lines.append(f'            Reliability Policy: Best Effort')
            lines.append(f'          Value: true')

    # TF
    lines.append('    - Class: rviz_default_plugins/TF')
    lines.append('      Enabled: true')
    lines.append('      Name: TF')
    lines.append('      Show Arrows: true')
    lines.append('      Show Axes: true')
    lines.append('      Show Names: true')
    lines.append('      Frame Timeout: 15')
    lines.append('      Value: true')

    # RobotModel per robot
    for ns, alias in zip(ROBOTS, ALIASES):
        lines.append(f'    - Class: rviz_default_plugins/RobotModel')
        lines.append(f'      Enabled: true')
        lines.append(f'      Name: {alias} Robot')
        lines.append(f'      Visual Enabled: true')
        lines.append(f'      Collision Enabled: false')
        lines.append(f'      Alpha: 1.0')
        lines.append(f'      Description Source: Topic')
        lines.append(f'      Robot Description Topic:')
        lines.append(f'        Value: /{ns}/robot_description')
        lines.append(f'        Depth: 5')
        lines.append(f'        Durability Policy: Transient Local')
        lines.append(f'        History Policy: Keep Last')
        lines.append(f'        Reliability Policy: Reliable')
        lines.append(f'      Value: true')

    lines.append('  Global Options:')
    lines.append('    Background Color: 48; 48; 48')
    lines.append('    Default Light: true')
    lines.append('    Fixed Frame: k1_0/pelvis')
    lines.append('    Frame Rate: 30')
    lines.append('  Global Clock:')
    lines.append('    Use sim time: false')
    lines.append('  Value: true')
    lines.append('  Views:')
    lines.append('    Current:')
    lines.append('      Class: rviz_default_plugins/Orbit')
    lines.append('      Distance: 3')
    lines.append('      Enable Stereo Rendering:')
    lines.append('        Stereo Eye Separation: 0.06')
    lines.append('        Stereo Focal Distance: 1')
    lines.append('        Swap Stereo Eyes: false')
    lines.append('        Value: false')
    lines.append('      Field of View: 0.7854')
    lines.append('      Focal Frame: k1_0/pelvis')
    lines.append('      Focal Shape Fixed Size: true')
    lines.append('      Focal Shape Size: 0.05')
    lines.append('      Invert Z Axis: false')
    lines.append('      Name: Current')
    lines.append('      Near Clip Distance: 0.01')
    lines.append('      Pitch: 0.5')
    lines.append('      Target Frame: k1_0/pelvis')
    lines.append('      Yaw: 0.785')
    lines.append('')
    lines.append('Window Geometry:')
    lines.append('  Displays:')
    lines.append('    collapsed: false')
    lines.append('  Height: 1080')
    lines.append('  Width: 1920')
    lines.append('  X: 0')
    lines.append('  Y: 0')

    return '\n'.join(lines)


if __name__ == '__main__':
    config = generate_config()
    out_dir = os.path.expanduser(
        '~/Projects/booster_ws/src/k1_fleet_rviz/config'
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'fleet_rviz.rviz')
    with open(out_path, 'w') as f:
        f.write(config)
    print(f'Written: {out_path}')
    print(f'Displays: 18 images + TF + 6 RobotModels = 25 total')

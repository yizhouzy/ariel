
"""Summary: Baby Controller.

Goal: Evolve a controller for the Baby robot that is able to:
1. Sense the battery level & switch to foraging mode when drops below threshold
2. Spin around to find the charging station using its camera
3. walk to the charging station and stop there

_Returns:
    _type_: _description_

_Raises:
    IndexError: _description_
"""
# Standard libraries
from typing import Literal, cast, List, Optional, Any
from pathlib import Path
import time
import os
import cv2
# Pretty little errors and progress bars
from rich.console import Console
from rich.traceback import install
# Initialize rich console and traceback handler
install()
console = Console()

# Third-party libraries
import numpy as np
import mujoco
import keyboard as kb
import matplotlib.pyplot as plt

# Network imports
import torch
from torch import nn
from torch.nn import Tanh

# Learner
from evotorch.algorithms import CMAES
from evotorch.neuroevolution import NEProblem

# Local libraries
from ariel.simulation.environments import SimpleFlatWorld
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.controllers.na_cpg import (NaCPG, create_fully_connected_adjacency)
from ariel.utils.tracker import Tracker
from ariel.simulation.controllers.utils.data_get import get_state_from_data as get_robot_state
from ariel.utils.renderers import VideoRecorder, video_renderer
from ariel.simulation.tasks.targeted_locomotion import (
    fitness_delta_distance,
    fitness_distance_and_efficiency,
    fitness_survival_and_locomotion,
    fitness_direct_path,
    distance_to_target,
    fitness_speed_to_target,
)
from baby_robot import baby_robot

# Set up command line argument parsing
# If none given, default values are used.
import argparse
parser = argparse.ArgumentParser(description='Evolution simulation with configurable budget')
parser.add_argument('--budget', type=int, default=40,
                    help='Number of generations for learning')
parser.add_argument('--dur', type=int, default=20,
                    help="Duration of an evaluation")
parser.add_argument('--population', type=int, default=10, help="Population size")
parser.add_argument('--fitness', type=str, default='distance', choices=['delta', 'efficiency', 'survival', 'direct', 'distance', 'speed'])
parser.add_argument('--reach-radius', type=float, default=0.25, help='Planar distance threshold for counting target arrival')
parser.add_argument('--num-actors', type=int, default=1, help='Number of parallel actors (CPUs) to use; set >1 to enable')
args = parser.parse_args()

BUDGET = args.budget
DURATION = args.dur
POP_SIZE = args.population
REACH_RADIUS = max(0.01, args.reach_radius)

# 1. Defined random target positions to prevent overfitting

TARGET_POSITIONS = [
    [-0.5 , -2, 0.1],  # Left
    [0.0, -2, 0.1],    # Center
    [0.5, -2, 0.1],  # Right
]

# Global constants
# Get file name and location to create data save folder.
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = Path(CWD / "__data__" / SCRIPT_NAME)
DATA.mkdir(exist_ok=True)


# ============================================================================ #
#                       Network and Helper function                            #
# ============================================================================ #
class Network(nn.Module):
    def __init__(
        self, input_size: int, output_size: int, hidden_size: int,
    ) -> None:
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

        self.hidden_activation = nn.ELU()
        self.output_activation = nn.Tanh()

        self.input = input_size

        # Disable gradients for all parameters
        for param in self.parameters():
            param.requires_grad = False

    @torch.inference_mode()
    def forward(self, model, data, state):
        x = torch.Tensor(state)

        x = self.hidden_activation(self.fc1(x))
        x = self.hidden_activation(self.fc2(x))
        x = self.output_activation(self.fc4(x)) * (torch.pi / 2)

        return x.detach().numpy()


@torch.no_grad()
def fill_parameters(net: nn.Module, vector: torch.Tensor) -> None:
    """Fill the parameters of a torch module (net) from a vector.

    No gradient information is kept.

    The vector's length must be exactly the same with the number
    of parameters of the PyTorch module.

    _Args:
        net: The torch module whose parameter values will be filled.
        vector: A 1-D torch tensor which stores the parameter values.

    _Raises:
        IndexError: If the vector is larger than expected.

    """
    address: int = 0
    for p in net.parameters():
        d = p.data.view(-1)
        n = len(d)
        d[:] = torch.as_tensor(vector[address : address + n], device=d.device)
        address += n

    if address != len(vector):
        error_msg = "The parameter vector is larger than expected"
        raise IndexError(error_msg)


# ============================================================================ #
#                         Camera frame processing                              #
# ============================================================================ #

def isolate_green(frame: np.ndarray) -> np.ndarray:
    # Convert to HSV color space
    # Enhances colours, makes it easier to detect target colour
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # Define range for green color (charging station marker)
    # Broadened range to be more robust to lab lighting; tune if needed
    lower_green = np.array([25, 30, 30])
    upper_green = np.array([95, 255, 255])

    # Create mask for green color
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    return green_mask


def analyze_sections(green_mask):
    sections = np.array_split(green_mask, 3, axis=1)
    left_section, middle_section, right_section = sections[0], sections[1], sections[2]

    # Calculate percentage of green pixels in each section
    def get_green_percentage(section):
        total_pixels = section.size
        if total_pixels == 0:
            return 0.0
        green_pixels = cv2.countNonZero(section)
        return (green_pixels / total_pixels)

    left_percent = get_green_percentage(left_section)
    middle_percent = get_green_percentage(middle_section)
    right_percent = get_green_percentage(right_section)

    return [left_percent, middle_percent, right_percent]


# Custom simulation runner with camera

def run_vision_simulation(
    model,
    data,
    network: Network,
    duration: int,
    target_position: Optional[np.ndarray] = None,
    renderer: Optional[mujoco.Renderer] = None,
    cam_name: Optional[str] = None,
    initial_battery: Optional[float] = None,
    control_step_freq: int = 50,
) -> dict[str, Any]:
    """Custom runner that processes vision.

    Returns runtime metrics including `final_battery` and `shaped_homing`.
    """

    # Setup Renderer if not passed (creates a new context)
    created_renderer = False
    if renderer is None:
        try:
            renderer = mujoco.Renderer(model, height=24, width=32)
            created_renderer = True
        except Exception:
            console.log("[yellow]run_vision_simulation: renderer init failed; running without vision.[/yellow]")
            renderer = None
            created_renderer = False

    timestep = model.opt.timestep

    # Initialize control placeholder
    current_action = np.zeros(model.nu)

    last_pos = np.array(data.qpos[0:3].copy())
    total_path_length = 0.0
    min_distance_to_target = float("inf")
    time_to_target: float | None = None

    trajectory = []

    # Battery: randomized per-episode unless a deterministic value is provided.
    if initial_battery is None:
        battery = float(np.random.rand())
    else:
        battery = float(initial_battery)
    battery_drain_per_sec = 1.0 / max(float(duration), 1.0)
    battery_decrement_per_step = battery_drain_per_sec * timestep

    # Shaped homing accumulator (accumulates positive reductions in planar distance
    # once the battery is low).
    shaped_homing = 0.0
    battery_threshold = 0.3
    prev_planar_distance: Optional[float] = None
    if target_position is not None:
        prev_planar_distance = float(np.linalg.norm(last_pos[:2] - np.asarray(target_position)[:2]))

    while data.time < duration:
        # Calculate deduced step count (Optimization from controller.py)
        deduced_step = int(np.ceil(data.time / timestep))

        # --- CONTROL STEP ---
        # Only run expensive vision and network pass every N steps
        if deduced_step % control_step_freq == 0:
            if renderer is not None:
                try:
                    renderer.update_scene(data, camera=cam_name)
                    img = renderer.render()
                    # 2. Process Vision
                    mask = isolate_green(img)
                    vision_inputs = analyze_sections(mask)
                except Exception:
                    # Renderer failed mid-run; fall back to zeros
                    vision_inputs = [0.0, 0.0, 0.0]
            else:
                vision_inputs = [0.0, 0.0, 0.0]

            # 3. Prepare Inputs
            robot_state = get_robot_state(data)

            # Using both sin and cos gives the network a smooth,
            # circular sense of time
            phase_inputs = [
                2 * np.sin(data.time * 2.0 * np.pi),
                2 * np.cos(data.time * 2.0 * np.pi),
            ]

            state_input = np.concatenate([
                robot_state,
                vision_inputs,
                phase_inputs,
                [battery],
            ]).astype(np.float32)

            # 4. Network Forward Pass
            current_action = network.forward(model, data, state_input)
            trajectory.append((data.qpos[0], data.qpos[1]))

        # 5. Apply Control (Hold previous action if not a control step)
        data.ctrl[:] = current_action

        # 6. Step Physics
        mujoco.mj_step(model, data)

        # Battery drains every physics step
        battery = max(0.0, battery - battery_decrement_per_step)

        current_pos = np.array(data.qpos[0:3].copy())
        total_path_length += np.linalg.norm(current_pos - last_pos)
        last_pos = current_pos

        if target_position is not None:
            planar_distance = float(np.linalg.norm(current_pos[:2]
                                                   - target_position[:2]))
            min_distance_to_target = min(
                                        min_distance_to_target,
                                        planar_distance)
            if time_to_target is None and planar_distance <= REACH_RADIUS:
                time_to_target = float(data.time)

            # Accumulate positive improvements in planar distance
            cur_planar_distance = planar_distance
            if battery <= battery_threshold:
                if prev_planar_distance is None:
                    prev_planar_distance = cur_planar_distance
                delta = prev_planar_distance - cur_planar_distance
                if delta > 0:
                    shaped_homing += delta
            prev_planar_distance = cur_planar_distance

    if target_position is None:
        min_distance_to_target = float(np.linalg.norm(last_pos[:2]))

    results = {
        "path_length": total_path_length,
        "trajectory": trajectory,
        "min_distance_to_target": min_distance_to_target,
        "time_to_target": time_to_target,
        "final_battery": battery,
        "shaped_homing": shaped_homing,
    }

    # Close any renderer we created locally to avoid lingering EGL contexts.
    if created_renderer:
        try:
            renderer.close()
        except Exception:
            pass

    return results


# ---------------------------------------------------------------------------
# Actor-local evaluator for parallel NEProblem actors
# This function will be pickled and executed on remote actors (via Ray).
# It lazily builds a local MuJoCo model/data and reuses it across evaluations
# inside the same actor process to avoid repeated compilation overhead.
# ---------------------------------------------------------------------------
_ACTOR_ENV: dict = {}

def _init_actor_env_once() -> None:
    if _ACTOR_ENV:
        return
    world = SimpleFlatWorld()

    # Add Charging Station + global camera (same as main)
    target_pos = TARGET_POSITIONS[0]
    target_body = world.spec.worldbody.add_body(name="charging_station",
                                                mocap=True,
                                                pos=target_pos)
    target_body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.1, 0.1, 0.1],
        rgba=[0, 1, 0, 1],
    )
    world.spec.worldbody.add_camera(
        name="video_cam",
        pos=[0, -1, 3],
        xyaxes=[1, 0, 0, 0, 3, 0],
    )

    # Spawn Baby
    baby_core = baby_robot()
    world.spawn(baby_core.spec, position=[0, 0, 0.1])

    model = world.spec.compile()
    data = mujoco.MjData(model)

    # Identify robot camera name
    robot_cam_name = None
    for i in range(model.ncam):
        name = model.camera(i).name
        if "camera" in name or "core" in name:
            robot_cam_name = name
            break
    if robot_cam_name is None and model.ncam > 0:
        robot_cam_name = model.camera(0).name

    # Pre-create a tiny renderer if available (vision), otherwise None
    try:
        renderer = mujoco.Renderer(model, height=24, width=32)
    except Exception:
        renderer = None

    # Mocap id for charging station
    try:
        target_mocap_id = model.body("charging_station").mocapid[0]
    except Exception:
        target_mocap_id = 0

    _ACTOR_ENV.update({
        "world": world,
        "model": model,
        "data": data,
        "robot_cam_name": robot_cam_name,
        "renderer": renderer,
        "target_mocap_id": target_mocap_id,
    })


def actor_fitness(net) -> float:
    """Top-level evaluator used by NEProblem in distributed mode.

    Receives a parameterised PyTorch module `net` and returns a scalar
    fitness. This mirrors the logic previously implemented in the
    `fitness_function` nested inside `evolve` but builds a local Mujoco
    environment per actor so evaluations are independent and parallel.
    """
    _init_actor_env_once()

    model = _ACTOR_ENV["model"]
    data = _ACTOR_ENV["data"]
    renderer = _ACTOR_ENV.get("renderer", None)
    robot_cam_name = _ACTOR_ENV.get("robot_cam_name", None)
    target_mocap_id = _ACTOR_ENV.get("target_mocap_id", 0)

    total_fitness = 0.0

    for target_pos in TARGET_POSITIONS:
        mujoco.mj_resetData(model, data)

        # Place target for this episode
        data.mocap_pos[target_mocap_id] = target_pos

        metrics = run_vision_simulation(
            model,
            data,
            network=net,
            duration=DURATION,
            target_position=np.asarray(target_pos),
            renderer=renderer,
            cam_name=robot_cam_name,
            control_step_freq=50,
        )

        final_pos = np.array(data.qpos[0:3].copy())
        final_z_height = final_pos[2]

        final_battery = metrics.get("final_battery", 0.0)
        battery_threshold = 0.3
        battery_low = final_battery <= battery_threshold

        if battery_low:
            if args.fitness == "delta":
                score = fitness_delta_distance(np.array([0.0, 0.0, 0.0]), final_pos, np.asarray(target_pos))
            elif args.fitness == "distance":
                score = distance_to_target(final_pos, np.asarray(target_pos))
            elif args.fitness == "survival":
                score = fitness_survival_and_locomotion(np.array([0.0, 0.0, 0.0]), final_pos, np.asarray(target_pos), final_z_height)
            elif args.fitness == "efficiency":
                score = fitness_distance_and_efficiency(np.array([0.0, 0.0, 0.0]), final_pos, np.asarray(target_pos), 0.0)
            elif args.fitness == "direct":
                score = fitness_direct_path(np.array([0.0, 0.0, 0.0]), final_pos, np.asarray(target_pos), metrics.get("path_length", 0.0))
            elif args.fitness == "speed":
                score = fitness_speed_to_target(
                    time_to_target=metrics.get("time_to_target", None),
                    duration=DURATION,
                    min_distance_to_target=metrics.get("min_distance_to_target", float("inf")),
                )
            else:
                score = distance_to_target(final_pos, np.asarray(target_pos))

            shaped = float(metrics.get("shaped_homing", 0.0))
            score = float(score) - shaped
        else:
            score = -float(metrics.get("path_length", 0.0))

        total_fitness += score

    return total_fitness / len(TARGET_POSITIONS)


# ============================================================================ #
#                         Define evolutionary loop                             #
# ============================================================================ #

def evolve(world, model, data) -> list[float]:

    """Evolve the robot's movement using an evolutionary algorithm."""
    tracker = Tracker(mujoco_obj_to_find=data, observable_attributes=["xpos"])
    tracker.setup(world.spec, data)

    # Identify Camera for ROBOT VISION (on the robot)
    robot_cam_name = None
    for i in range(model.ncam):
        name = model.camera(i).name
        if "camera" in name or "core" in name:
            robot_cam_name = name
            break
    if robot_cam_name is None and model.ncam > 0:
        robot_cam_name = model.camera(0).name

    # Pre-initialize renderer for the evolution loop (best-effort).
    # In headless environments renderer creation may fail; fall back to None.
    try:
        renderer = mujoco.Renderer(model, height=24, width=32)
    except Exception:
        console.log("[yellow]Renderer init failed; running evolution without pre-created renderer.[/yellow]")
        renderer = None

    # Get Mocap ID for the charging station
    try:
        target_mocap_id = model.body("charging_station").mocapid[0]
    except:
        console.print("[red]Error: Charging station mocap body not found![/red]")
        target_mocap_id = 0

    # Define the fitness function
    def fitness_function(x: Network) -> float:
        
        total_fitness = 0.0

        for target_pos in TARGET_POSITIONS:
            mujoco.mj_resetData(model, data)
            tracker.reset()

            data.mocap_pos[target_mocap_id] = target_pos

            # 1. Capture Initial State BEFORE simulation
            initial_pos = np.array(data.qpos[0:3].copy())
            target_pos_arr = np.array(target_pos)

            # Run Simulation
            metrics = run_vision_simulation(
                model,
                data,
                network=x,
                duration=DURATION,
                target_position=target_pos_arr,
                renderer=renderer,
                cam_name=robot_cam_name,
                control_step_freq=50,
            )

            # 2. Capture Final State AFTER simulation
            final_pos = np.array(data.qpos[0:3].copy())
            final_z_height = final_pos[2]

            # Conditional fitness: only reward homing when battery is low
            final_battery = metrics.get("final_battery", 0.0)
            battery_threshold = 0.3
            battery_low = final_battery <= battery_threshold

            if battery_low:
                # When battery low, reward homing according to selected fitness
                if args.fitness == "delta":
                    score = fitness_delta_distance(initial_pos,
                                                   final_pos,
                                                   target_pos_arr)
                elif args.fitness == "distance":
                    score = distance_to_target(final_pos, target_pos_arr)
                elif args.fitness == "survival":
                    score = fitness_survival_and_locomotion(initial_pos,
                                                            final_pos,
                                                            target_pos_arr,
                                                            final_z_height)
                elif args.fitness == "efficiency":
                    score = fitness_distance_and_efficiency(initial_pos,
                                                            final_pos,
                                                            target_pos_arr,
                                                            0.0)
                elif args.fitness == "direct":
                    score = fitness_direct_path(initial_pos,
                                                final_pos,
                                                target_pos_arr,
                                                metrics["path_length"])
                elif args.fitness == "speed":
                    score = fitness_speed_to_target(
                        time_to_target=metrics["time_to_target"],
                        duration=DURATION,
                        min_distance_to_target=metrics["min_distance_to_target"],
                    )
                else:
                    score = distance_to_target(final_pos, target_pos_arr)
                # Apply shaped homing bonus (reduces score since lower is better)
                shaped = float(metrics.get("shaped_homing", 0.0))
                score = float(score) - shaped
            else:
                # Battery not low -> prioritize locomotion/exploration
                # (longer path is better)
                score = -float(metrics.get("path_length", 0.0))

            total_fitness += score

        return total_fitness / len(TARGET_POSITIONS)

    console.log(f"Evolving for {BUDGET} generations with Vision Input")

    # --- CALCULATE NEW INPUT SIZE ---
    num_joints = len(data.qpos) - 7

    # Inputs: Quat(3) + Joints(N) + Vision(3) + Heartbeat(2) + Battery(1)
    # In case of the spider: 3 + 8 + 3 + 2 + 1
    input_dim = 3 + num_joints + 3 + 2 + 1

    # Initialise Neural Network Controller
    # 14 input neurons
    # 32 hidden layer neurons
    # 8 output neurons
    network = Network(
        input_size=input_dim,
        output_size=model.nu,
        hidden_size=32,
    )

    # Initialise Problem for the solver/learner
    # Enable distributed actors only when the user requests more than 1 actor
    num_actors_cfg = args.num_actors if (hasattr(args, "num_actors") 
                                         and args.num_actors is not None 
                                         and args.num_actors > 1) else None
    if num_actors_cfg is not None:
        # Ask Ray to avoid packaging large or irrelevant folders to speed startup
        actor_config = {
            "num_cpus": 1,
            "runtime_env": {
                "excludes": [
                    ".git",
                    ".venv",
                    "__pycache__",
                ]
            },
        }
    else:
        actor_config = None

    problem = NEProblem(
            objective_sense="min",
            network_eval_func=actor_fitness,
            network=network.eval(),
            initial_bounds=(-0.5, 0.5),
            device="cpu",
            num_actors=num_actors_cfg,
            actor_config=actor_config,
    )

    # Initialise CMA-ES learner
    searcher = CMAES(problem=problem,
                     stdev_init=0.075,
                     popsize=POP_SIZE,
                     )

    console.log(f"Population size: {searcher.popsize}")

    for bud in range(BUDGET + 1):
        searcher.step()
        gen_best = searcher.status["pop_best_eval"]

        console.rule(f"Budget: {bud}/{BUDGET}")
        console.log(f"Best Fit (Avg): {gen_best:.4f}")

    # Close the renderer pre-initialized for evolution to free EGL resources
    try:
        renderer.close()
    except Exception:
        pass

    best_ind = searcher.status["best"].values
    return best_ind, input_dim


# ============================================================================ #
#                           Main entry function                                #
# ============================================================================ #

def main():
    mujoco.set_mjcb_control(None)

    # Initialise world
    world = SimpleFlatWorld()

    # Add Charging Station Object (mocap body)
    target_pos = TARGET_POSITIONS[0]
    target_body = world.spec.worldbody.add_body(name="charging_station",
                                                mocap=True,
                                                pos=target_pos)
    target_body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.1, 0.1, 0.1],
        rgba=[0, 1, 0, 1],
    )

    # Global Camera for Video Recording
    world.spec.worldbody.add_camera(
        name="video_cam",
        pos=[0, -1, 3],
        xyaxes=[1, 0, 0, 0, 3, 0],
    )

    baby_core = baby_robot()
    world.spawn(baby_core.spec, position=[0, 0, 0.1])

    model = world.spec.compile()
    data = mujoco.MjData(model)

    best_weights, final_input_dim = evolve(world, model, data)

    return model, data, best_weights, world, final_input_dim


if __name__ == "__main__":
    start = time.time()
    model, data, best_weights, world, input_dim = main()
    end = time.time()

    console.log(f"Evolution took {(end - start) / 60:.2f} minutes")

    weights_path = "3_baby_vision_new.npy"
    # Unconditionally save the new weights, overwriting any old ones
    np.save(weights_path, best_weights)
    console.log(f"Best weights saved to {weights_path}")

# ============================================================================ #
#                           Initialise world and                               #
#                           load best  performer                               #
#                           for  video recording                               #
# ============================================================================ #
    network = Network(
        input_size=input_dim,
        output_size=model.nu,
        hidden_size=32,
    )
    fill_parameters(network, torch.Tensor(best_weights))

    # Identify robot camera
    robot_cam_name = None
    for i in range(model.ncam):
        cam_name = model.camera(i).name
        # Check for 'core' to match the spider's camera naming convention
        if ("camera" in cam_name or "core" in cam_name) and "video" not in cam_name:
            robot_cam_name = cam_name
            break

    console.log("Rendering Best Video...")
    path_to_video_folder = str(DATA / "videos")

    mujoco.mj_resetData(model, data)

    # Set target to middle position for the video demo
    target_mocap_id = model.body("charging_station").mocapid[0]
    data.mocap_pos[target_mocap_id] = TARGET_POSITIONS[0]

    # 1. Renderer for Robot Vision (Low Res)
    try:
        control_renderer = mujoco.Renderer(model, height=24, width=32)
    except Exception:
        console.log("[yellow]control_renderer init failed; replay will run without low-res vision.[/yellow]")
        control_renderer = None

    # 2. Renderer for Video Output (High Res)
    # (use context-managed renderer below; avoid creating an extra EGL context here)

    def get_vision_control_signal(m, d):
        if robot_cam_name and control_renderer is not None:
            try:
                control_renderer.update_scene(d, camera=robot_cam_name)
                img = control_renderer.render()
                mask = isolate_green(img)
                vision_inputs = analyze_sections(mask)
            except Exception:
                vision_inputs = [0, 0, 0]
        else:
            vision_inputs = [0, 0, 0]

        robot_state = get_robot_state(d)

        phase_inputs = [
            2 * np.sin(d.time * 2.0 * np.pi),
            2 * np.cos(d.time * 2.0 * np.pi),
        ]

        # Approximate battery for replay: drains linearly with time
        battery = max(0.0, 1.0 - (d.time / max(float(DURATION), 1.0)))

        state = np.concatenate([
            robot_state,
            vision_inputs,
            phase_inputs,
            [battery],
        ]).astype(np.float32)

        return network.forward(m, d, state)

    # Video Timing Variables
    fps = 30
    video_dt = 1.0 / fps
    next_video_time = 0.0
    frames = []

    # Simulation Timing Variables
    # (must match evolve/run_vision_simulation exactly)
    timestep = model.opt.timestep
    control_step_freq = 50

    current_ctrl = np.zeros(model.nu)

# --- REPLAY BEST & RECORD VIDEO ---
    console.log("Rendering Best Video...")
    path_to_video_folder = str(DATA / "videos")

    # 1. Setup VideoRecorder (using your Ariel library class)
    video_recorder = VideoRecorder(
        file_name="baby_vision_best",
        output_folder=path_to_video_folder,
    )

    # 2. Reset Simulation & Target
    mujoco.mj_resetData(model, data)
    target_mocap_id = model.body("charging_station").mocapid[0]
    data.mocap_pos[target_mocap_id] = TARGET_POSITIONS[0]

    # 3. Setup Visualization Options (from your snippet)
    viz_options = mujoco.MjvOption()
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = False
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_BODYBVH] = False

    # 4. Get Camera ID ("video_cam" is the one we created earlier)
    camera_id = mujoco.mj_name2id(model,
                                  mujoco.mjtObj.mjOBJ_CAMERA,
                                  "video_cam")

    # 5. Timing Variables
    fps = 30
    dt = model.opt.timestep
    steps_per_frame = int(1.0 / (fps * dt))
    control_step_freq = 50
    current_ctrl = np.zeros(model.nu)

    # 6. Setup separate renderer for the Robot's Vision (Low Res)
    # We keep this outside the video loop so we don't recreate it every frame

    # 7. Main Rendering Loop (Using Context Manager as requested)
    # We use the video_recorder width/height for the output video. Attempt to
    # create a high-res renderer and run recording; if unavailable (headless)
    # skip video rendering but still run a deterministic evaluation for plotting.
    try:
        tmp_renderer = mujoco.Renderer(model, height=480, width=640)
    except Exception:
        tmp_renderer = None

    if tmp_renderer is not None:
        with tmp_renderer as renderer:
            while data.time < DURATION:
                # INNER LOOP: Step physics N times to match Video FPS
                for _ in range(steps_per_frame):
                    deduced_step = int(np.ceil(data.time / dt))
                    if deduced_step % control_step_freq == 0:
                        current_ctrl = get_vision_control_signal(model, data)

                    data.ctrl[:] = current_ctrl
                    mujoco.mj_step(model, data)

                renderer.update_scene(
                    data,
                    scene_option=viz_options,
                    camera=camera_id,
                )
                video_recorder.write(frame=renderer.render())

        # Finish recording
        video_recorder.release()
        console.log(f"[green]Video rendering complete. Saved to {path_to_video_folder}[/green]")

    else:
        console.log("[yellow]High-res renderer unavailable; skipped video rendering.[/yellow]")

    # 9. Save Path Figure (run a deterministic evaluation for the plot)
    test_target = TARGET_POSITIONS[0]
    mujoco.mj_resetData(model, data)
    data.mocap_pos[target_mocap_id] = test_target

    metrics = run_vision_simulation(
        model,
        data,
        network=network,
        duration=DURATION,
        target_position=np.asarray(test_target),
        renderer=None,
        cam_name=robot_cam_name,
        initial_battery=1.0,
        control_step_freq=50,
    )

    # Extract X and Y coordinates
    path = metrics["trajectory"]
    x_coords = [p[0] for p in path]
    y_coords = [p[1] for p in path]

    # Create the plot
    plt.figure(figsize=(8, 8))
    plt.plot(x_coords[0], y_coords[0], "go", markersize=10, label="Start")
    plt.plot(test_target[0], test_target[1], "r*", markersize=15, label="Target")
    plt.plot(x_coords, y_coords, "b-", linewidth=2, label="Robot Path")
    plt.title(f"Robot Trajectory Map (Fitness: {args.fitness})")
    plt.xlabel("X Position (meters)")
    plt.ylabel("Y Position (meters)")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(path_to_video_folder, f"trajectory_{args.fitness}.png")
    plt.savefig(plot_path)
    console.log(f"[green]Trajectory map saved to {plot_path}[/green]")

    # Cleanup control renderer used for low-res vision
    try:
        control_renderer.close()
    except Exception:
        pass

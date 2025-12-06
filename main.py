from config import get_default_config, Config
from model import build_model
from helpers import set_global_seed, save_checkpoint
from train import train_all
from evaluate import run_full_evaluation
from sampling import make_sampling_sets

def main():
    # Config and seed
    cfg: Config = get_default_config()
    set_global_seed(cfg.training.seed)

    # Build model
    model = build_model(cfg.network)

    # Train over all stages
    model, optimizer, history = train_all(model, cfg)

    # Save final checkpoint
    save_checkpoint(model, optimizer, cfg.training.checkpoint_dir)

    # Plotting coordinate cloud
    final_stage_cfg = cfg.training.stages[-1]
    plot_coords = make_sampling_sets(
        final_stage_cfg, cfg.geometry
        )

    # Evaluate and make plots
    metrics = run_full_evaluation(
        model=model,
        cfg=cfg,
        out_dir="figs",
        deformed_scale=1.0,  # adjust if you want more/less exaggeration
        history=history,
        coords=plot_coords
    )

    print("Final tip displacement and error:")
    print(f"  u_tip_x = {metrics['u_tip_x']:.6f}")
    print(f"  u_tip_y = {metrics['u_tip_y']:.6f}")
    print(f"  tip error = {metrics['tip_error_percent']:.3f}%")

if __name__ == "__main__":
    main()

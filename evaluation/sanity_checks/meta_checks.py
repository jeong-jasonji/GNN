import numpy as np
from evaluation.sanity_checks.logger import DiagnosticsLogger
from evaluation.sanity_checks.meta_visuals import plot_prototypes, plot_meta_adaptation

def meta_diagnostics(model, episodic_loader, log_dir, step=0, n_episodes=5):
    """Run episodic evaluations and log meta-learning metrics."""
    logger = DiagnosticsLogger(log_dir)
    meta_losses, meta_accs = [], []
    proto_fig = None

    for i in range(n_episodes):
        episode = episodic_loader.sample_episode()
        loss, acc, *extras = model.meta_train_step(episode)
        meta_losses.append(loss.numpy() if hasattr(loss, "numpy") else float(loss))
        meta_accs.append(acc.numpy() if hasattr(acc, "numpy") else float(acc))

        if "protonet" in model.__class__.__name__.lower() and len(extras) >= 3:
            s_x, s_y, q_x, q_y, prototypes = extras
            proto_fig = plot_prototypes(s_x, s_y, q_x, q_y, prototypes)

    logger.log_scalar("meta/mean_episode_loss", np.mean(meta_losses), step)
    logger.log_scalar("meta/mean_episode_accuracy", np.mean(meta_accs), step)
    if proto_fig:
        logger.log_figure("meta/prototype_distribution", proto_fig, step)

    loss_fig = plot_meta_adaptation(meta_losses)
    logger.log_figure("meta/adaptation_loss_curve", loss_fig, step)
    logger.close()
    print(f"[MetaDiagnostics] Episodes={n_episodes}, MeanLoss={np.mean(meta_losses):.4f}, MeanAcc={np.mean(meta_accs):.4f}")

import matplotlib.pyplot as plt
import plotly.graph_objects as go


def plot_metric_by_frame(df, metric, ylabel, title, color_by="skip"):
    """Reusable Matplotlib plot for any frame-based metric."""
    clean_df = df.dropna()
    plt.figure()
    plt.plot(clean_df['frame'], clean_df[metric], color='gray', linestyle='-', zorder=1)
    scatter = plt.scatter(clean_df['frame'], clean_df[metric], c=clean_df[color_by], cmap='viridis', marker='o', zorder=2)
    plt.title(title)
    plt.xlabel("Frame Index")
    plt.ylabel(ylabel)
    cbar = plt.colorbar(scatter)
    cbar.set_label(color_by.replace('_', ' ').title())
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_single_matcher_metrics(df, matcher_name):
    plot_metric_by_frame(df, 'translation_error', 'Translation Error (degrees)', f"Translation Error - {matcher_name}")
    plot_metric_by_frame(df, 'rotation_error', 'Rotation Error (degrees)', f"Rotation Error - {matcher_name}")
    plot_metric_by_frame(df, 'reprojection_error', 'Reprojection Error (pixels)', f"Reprojection Error - {matcher_name}")


def plot_combined_matcher_comparison(all_metrics):
    all_df = pd.concat(all_metrics, ignore_index=True).dropna()
    for metric, ylabel in [
        ('rotation_error', 'Rotation Error (degrees)'),
        ('translation_error', 'Translation Error (degrees)'),
        ('skip', 'Number of Skips'),
        ('reprojection_error', 'Reprojection Error (pixels)')
    ]:
        plt.figure()
        for matcher in all_df['matcher'].unique():
            subset = all_df[all_df['matcher'] == matcher]
            plt.plot(subset['frame'], subset[metric], label=matcher)
        plt.title(f"{ylabel} Comparison Across Matchers")
        plt.xlabel("Frame Index")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def plot_trajectory_plotly(positions, title="Estimated Camera Trajectory"):
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
        mode='lines+markers',
        marker=dict(size=4, color='red'),
        line=dict(color='blue', width=2),
        name="Camera Path"
    ))

    # Mark the start position in green
    fig.add_trace(go.Scatter3d(
        x=[positions[0, 0]], y=[positions[0, 1]], z=[positions[0, 2]],
        mode='markers',
        marker=dict(size=6, color='green', symbol='circle'),
        name="Start"
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()

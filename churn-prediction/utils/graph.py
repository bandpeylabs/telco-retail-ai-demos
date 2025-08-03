"""
Graph visualization utilities for NetworkX graphs using PyVis.

This module provides enhanced graph visualization capabilities with customizable
styling, centrality-based coloring, and interactive features for network analysis.
"""

from pyvis.network import Network
import networkx as nx
import os
import uuid
import shutil
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class GraphVisualizer:
    """
    Enhanced graph visualizer with customizable styling and centrality analysis.

    This class provides methods to create rich, interactive visualizations
    of NetworkX graphs with various styling options and centrality-based
    node coloring.
    """

    def __init__(self, height: str = "750px", width: str = "100%",
                 directed: bool = True, notebook: bool = True):
        """
        Initialize the graph visualizer.

        Args:
            height: Height of the visualization container
            width: Width of the visualization container
            directed: Whether the graph is directed
            notebook: Whether running in notebook environment
        """
        self.height = height
        self.width = width
        self.directed = directed
        self.notebook = notebook

        # Default color schemes
        self.color_schemes = {
            'viridis': ['#440154', '#31688E', '#35B779', '#FDE725'],
            'plasma': ['#0D0887', '#7E03A8', '#CC4778', '#F89441'],
            'inferno': ['#000004', '#1B0F41', '#8A2268', '#E95B1A'],
            'magma': ['#000004', '#1A1043', '#51127C', '#B63679'],
            'cool': ['#003F5C', '#2E4A7B', '#665191', '#A05195'],
            'warm': ['#FF6B35', '#F7931E', '#FFD23F', '#FEFE56']
        }

    def create_network(self, graph: nx.Graph,
                       node_size_attr: Optional[str] = None,
                       node_color_attr: Optional[str] = None,
                       edge_weight_attr: Optional[str] = None,
                       color_scheme: str = 'viridis',
                       show_labels: bool = True,
                       label_threshold: float = 0.1) -> Network:
        """
        Create a PyVis network from a NetworkX graph with enhanced styling.

        Args:
            graph: NetworkX graph to visualize
            node_size_attr: Node attribute to use for sizing (default: degree)
            node_color_attr: Node attribute to use for coloring (default: pagerank)
            edge_weight_attr: Edge attribute to use for edge thickness
            color_scheme: Color scheme to use for node coloring
            show_labels: Whether to show node labels
            label_threshold: Threshold for showing labels (top percentile)

        Returns:
            PyVis Network object
        """
        # Create PyVis network
        net = Network(
            height=self.height,
            width=self.width,
            directed=self.directed,
            cdn_resources='remote',
            notebook=self.notebook
        )

        # Set physics options for better layout
        net.set_options("""
        var options = {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "maxVelocity": 50,
            "minVelocity": 0.1,
            "solver": "forceAtlas2Based",
            "timestep": 0.35
          }
        }
        """)

        # Add nodes with styling
        self._add_nodes_with_styling(net, graph, node_size_attr, node_color_attr,
                                     color_scheme, show_labels, label_threshold)

        # Add edges with styling
        self._add_edges_with_styling(net, graph, edge_weight_attr)

        return net

    def _add_nodes_with_styling(self, net: Network, graph: nx.Graph,
                                node_size_attr: Optional[str],
                                node_color_attr: Optional[str],
                                color_scheme: str, show_labels: bool,
                                label_threshold: float):
        """Add nodes to the network with enhanced styling."""

        # Calculate node attributes if not provided
        if node_size_attr is None:
            node_sizes = dict(nx.degree_centrality(graph))
        else:
            node_sizes = nx.get_node_attributes(graph, node_size_attr)
            if not node_sizes:
                node_sizes = dict(nx.degree_centrality(graph))

        if node_color_attr is None:
            node_colors = nx.pagerank(graph, alpha=0.85)
        else:
            node_colors = nx.get_node_attributes(graph, node_color_attr)
            if not node_colors:
                node_colors = nx.pagerank(graph, alpha=0.85)

        # Normalize values for visualization
        size_values = list(node_sizes.values())
        color_values = list(node_colors.values())

        if size_values:
            min_size, max_size = min(size_values), max(size_values)
            size_range = max_size - min_size if max_size != min_size else 1
        else:
            min_size, max_size, size_range = 0, 1, 1

        if color_values:
            min_color, max_color = min(color_values), max(color_values)
            color_range = max_color - min_color if max_color != min_color else 1
        else:
            min_color, max_color, color_range = 0, 1, 1

        # Determine which nodes to label
        if show_labels and size_values:
            threshold_value = np.percentile(
                size_values, (1 - label_threshold) * 100)
        else:
            threshold_value = float('inf')

        # Add nodes with styling
        for node in graph.nodes():
            # Calculate normalized size (10-50 range)
            size = node_sizes.get(node, 0)
            normalized_size = 10 + 40 * (size - min_size) / size_range

            # Calculate color
            color_val = node_colors.get(node, 0)
            normalized_color = (color_val - min_color) / color_range

            # Get color from scheme
            color = self._get_color_from_scheme(normalized_color, color_scheme)

            # Node title for tooltip
            title = f"Node: {node}<br>"
            title += f"Size Metric: {size:.3f}<br>"
            title += f"Color Metric: {color_val:.3f}"

            # Add node
            net.add_node(
                node,
                label=str(node) if size >= threshold_value else "",
                size=normalized_size,
                color=color,
                title=title,
                borderWidth=2,
                borderColor="#2B2B2B"
            )

    def _add_edges_with_styling(self, net: Network, graph: nx.Graph,
                                edge_weight_attr: Optional[str]):
        """Add edges to the network with enhanced styling."""

        # Get edge weights if specified
        if edge_weight_attr:
            edge_weights = nx.get_edge_attributes(graph, edge_weight_attr)
        else:
            edge_weights = {}

        # Normalize edge weights
        if edge_weights:
            weight_values = list(edge_weights.values())
            min_weight, max_weight = min(weight_values), max(weight_values)
            weight_range = max_weight - min_weight if max_weight != min_weight else 1
        else:
            min_weight, max_weight, weight_range = 0, 1, 1

        # Add edges
        for edge in graph.edges():
            source, target = edge

            # Calculate edge width (1-5 range)
            weight = edge_weights.get(edge, 1)
            normalized_weight = 1 + 4 * (weight - min_weight) / weight_range

            # Edge title for tooltip
            title = f"Edge: {source} → {target}<br>"
            title += f"Weight: {weight:.3f}"

            net.add_edge(
                source,
                target,
                width=normalized_weight,
                color="#666666",
                title=title,
                arrows="to" if self.directed else None
            )

    def _get_color_from_scheme(self, normalized_value: float, scheme: str) -> str:
        """Get color from color scheme based on normalized value."""
        if scheme not in self.color_schemes:
            scheme = 'viridis'

        colors = self.color_schemes[scheme]
        n_colors = len(colors)

        # Interpolate between colors
        index = normalized_value * (n_colors - 1)
        low_index = int(index)
        high_index = min(low_index + 1, n_colors - 1)
        fraction = index - low_index

        # Simple linear interpolation (in production, use proper color interpolation)
        if low_index == high_index:
            return colors[low_index]
        else:
            return colors[low_index]  # Simplified for now

    def display_graph(self, graph: nx.Graph,
                      title: str = "Network Graph",
                      node_size_attr: Optional[str] = None,
                      node_color_attr: Optional[str] = None,
                      edge_weight_attr: Optional[str] = None,
                      color_scheme: str = 'viridis',
                      show_labels: bool = True,
                      label_threshold: float = 0.1) -> Tuple[str, str]:
        """
        Display a graph with enhanced styling and return HTML and file path.

        Args:
            graph: NetworkX graph to visualize
            title: Title for the visualization
            node_size_attr: Node attribute for sizing
            node_color_attr: Node attribute for coloring
            edge_weight_attr: Edge attribute for thickness
            color_scheme: Color scheme to use
            show_labels: Whether to show node labels
            label_threshold: Threshold for showing labels

        Returns:
            Tuple of (HTML content, file path)
        """
        # Create network
        net = self.create_network(
            graph=graph,
            node_size_attr=node_size_attr,
            node_color_attr=node_color_attr,
            edge_weight_attr=edge_weight_attr,
            color_scheme=color_scheme,
            show_labels=show_labels,
            label_threshold=label_threshold
        )

        # Generate unique file names
        temp_html_path = f"/tmp/{uuid.uuid4().hex}.html"
        download_path = f"/dbfs/FileStore/{uuid.uuid4().hex}.html"

        # Show and save
        net.show(temp_html_path)

        # Move to accessible location
        try:
            shutil.move(temp_html_path, download_path)
        except Exception as e:
            print(f"Warning: Could not move file to {download_path}: {e}")
            download_path = temp_html_path

        # Enhance HTML with custom styling
        enhanced_html = self._enhance_html(net.html, title)

        return enhanced_html, download_path

    def _enhance_html(self, html_content: str, title: str) -> str:
        """Enhance HTML with custom styling and title."""
        # Add custom CSS and title
        enhanced_head = f'''
        <head>
            <title>{title}</title>
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css" type="text/css"/>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }}
                .container {{
                    background: white;
                    border-radius: 10px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                    padding: 20px;
                    margin: 0 auto;
                    max-width: 1200px;
                }}
                h1 {{
                    color: #333;
                    text-align: center;
                    margin-bottom: 20px;
                    font-weight: 300;
                }}
                .graph-container {{
                    border: 1px solid #e0e0e0;
                    border-radius: 8px;
                    overflow: hidden;
                }}
            </style>
        </head>
        '''

        # Replace the head section
        html_content = html_content.replace('<head>', enhanced_head)

        # Wrap content in container
        body_start = html_content.find('<body>')
        body_end = html_content.find('</body>')

        if body_start != -1 and body_end != -1:
            body_content = html_content[body_start + 7:body_end]
            new_body = f'''
            <body>
                <div class="container">
                    <h1>{title}</h1>
                    <div class="graph-container">
                        {body_content}
                    </div>
                </div>
            </body>
            '''
            html_content = html_content.replace(
                html_content[body_start:body_end + 8], new_body)

        return html_content


def displayGraph(graph: nx.Graph, **kwargs) -> Tuple[str, str]:
    """
    Legacy function for backward compatibility.

    Args:
        graph: NetworkX graph to visualize
        **kwargs: Additional arguments passed to GraphVisualizer

    Returns:
        Tuple of (HTML content, file path)
    """
    visualizer = GraphVisualizer()
    return visualizer.display_graph(graph, **kwargs)


def create_centrality_visualization(graph: nx.Graph,
                                    centrality_type: str = 'pagerank',
                                    title: str = None) -> Tuple[str, str]:
    """
    Create a specialized visualization for centrality analysis.

    Args:
        graph: NetworkX graph
        centrality_type: Type of centrality to visualize ('pagerank', 'degree', 'betweenness', 'closeness')
        title: Custom title for the visualization

    Returns:
        Tuple of (HTML content, file path)
    """
    visualizer = GraphVisualizer()

    # Set default title if not provided
    if title is None:
        title = f"Network Graph - {centrality_type.title()} Centrality"

    # Map centrality types to node attributes
    centrality_mapping = {
        'pagerank': 'pagerank',
        'degree': 'degree',
        'betweenness': 'betweenness',
        'closeness': 'closeness'
    }

    node_color_attr = centrality_mapping.get(centrality_type, 'pagerank')

    return visualizer.display_graph(
        graph=graph,
        title=title,
        node_color_attr=node_color_attr,
        color_scheme='viridis',
        show_labels=True,
        label_threshold=0.1
    )


def create_community_visualization(graph: nx.Graph,
                                   communities: Dict[int, int],
                                   title: str = "Network Graph - Community Structure") -> Tuple[str, str]:
    """
    Create a specialized visualization for community structure.

    Args:
        graph: NetworkX graph
        communities: Dictionary mapping node IDs to community IDs
        title: Title for the visualization

    Returns:
        Tuple of (HTML content, file path)
    """
    visualizer = GraphVisualizer()

    # Add community information to graph
    nx.set_node_attributes(graph, communities, 'community')

    return visualizer.display_graph(
        graph=graph,
        title=title,
        node_color_attr='community',
        color_scheme='cool',
        show_labels=True,
        label_threshold=0.15
    )

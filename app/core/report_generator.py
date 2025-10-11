from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core import storage
from app.core.checkpoints import FolderCheckpoint
from app.core.currency import get_usd_to_inr, convert_usd_to_inr


def generate_project_report(
    project_id: int, 
    checkpoint_dir: Optional[Path] = None
) -> str:
    """Generate comprehensive markdown report for a project.
    
    Combines data from:
    - Database runs for this project
    - Checkpoint data if checkpoint_dir provided
    - Live currency conversion rates
    
    Args:
        project_id: Project ID to generate report for
        checkpoint_dir: Optional path to folder with checkpoint data
    
    Returns:
        Markdown formatted report string
    """
    # Get project info
    project = storage.get_project_by_id(project_id)
    if not project:
        return f"# Error: Project {project_id} not found\n"
    
    # Get database stats
    db_stats = storage.get_project_stats(project_id)
    
    # Get checkpoint stats if available
    checkpoint_stats = None
    if checkpoint_dir and checkpoint_dir.exists():
        try:
            cp = FolderCheckpoint(checkpoint_dir)
            cp.load()
            checkpoint_stats = cp.get_processing_stats()
        except Exception:
            checkpoint_stats = None
    
    # Get currency rate
    usd_to_inr = get_usd_to_inr()
    
    # Build report
    report_lines: List[str] = []
    
    # Header
    report_lines.append(f"# Project Report: {project.name}")
    report_lines.append(f"\n**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"\n**Project Description:** {project.description or 'N/A'}")
    report_lines.append("\n---\n")
    
    # Summary Section
    report_lines.append("## Summary")
    report_lines.append("")
    
    # Use checkpoint stats if available, otherwise database stats
    if checkpoint_stats and checkpoint_stats.get("total_images", 0) > 0:
        total_images = checkpoint_stats.get("total_images", 0)
        total_cost = checkpoint_stats.get("total_cost_usd", 0.0)
        total_tokens_in = checkpoint_stats.get("total_tokens_input", 0)
        total_tokens_out = checkpoint_stats.get("total_tokens_output", 0)
        images_processed = checkpoint_stats.get("images_processed", 0)
        images_failed = checkpoint_stats.get("images_failed", 0)
    else:
        total_images = db_stats.get("total_images", 0)
        total_cost = db_stats.get("total_cost_usd", 0.0)
        total_tokens_in = 0  # Not tracked in DB
        total_tokens_out = 0  # Not tracked in DB
        images_processed = total_images
        images_failed = 0
    
    report_lines.append(f"- **Total Images Processed:** {images_processed}")
    if images_failed > 0:
        report_lines.append(f"- **Images Failed:** {images_failed}")
    
    if total_tokens_in > 0 or total_tokens_out > 0:
        report_lines.append(f"- **Total Tokens (Input):** {total_tokens_in:,}")
        report_lines.append(f"- **Total Tokens (Output):** {total_tokens_out:,}")
        report_lines.append(f"- **Total Tokens:** {total_tokens_in + total_tokens_out:,}")
    
    report_lines.append("")
    
    # Cost Section
    report_lines.append("## Cost Analysis")
    report_lines.append("")
    report_lines.append(f"- **Total Cost (USD):** ${total_cost:.6f}")
    
    if usd_to_inr:
        inr_cost = convert_usd_to_inr(total_cost, rate=usd_to_inr)
        if inr_cost is not None:
            report_lines.append(f"- **Total Cost (INR):** ₹{inr_cost:.2f}")
            report_lines.append(f"- **Exchange Rate:** 1 USD = ₹{usd_to_inr:.2f}")
    
    if images_processed > 0:
        avg_cost = total_cost / images_processed
        report_lines.append(f"- **Average Cost per Image (USD):** ${avg_cost:.6f}")
        if usd_to_inr:
            avg_inr = convert_usd_to_inr(avg_cost, rate=usd_to_inr)
            if avg_inr is not None:
                report_lines.append(f"- **Average Cost per Image (INR):** ₹{avg_inr:.4f}")
    
    report_lines.append("")
    
    # Models Used Section
    models_used = db_stats.get("models_used", [])
    if models_used:
        report_lines.append("## Models Used")
        report_lines.append("")
        for model in models_used:
            report_lines.append(f"- {model}")
        report_lines.append("")
    
    # Model Pricing Section (get from first run)
    with storage.get_db() as db:
        first_run = (
            db.query(storage.Run)
            .filter(storage.Run.project_id == project_id)
            .order_by(storage.Run.created_at.asc())
            .first()
        )
        
        if first_run and first_run.provider:
            provider = first_run.provider
            catalog_caps = provider.catalog_caps_json or {}
            
            # Try to extract pricing info
            cost_block = catalog_caps.get("cost") or catalog_caps.get("pricing")
            if cost_block and isinstance(cost_block, dict):
                report_lines.append("## Model Pricing")
                report_lines.append("")
                
                # Handle different pricing formats
                if "input" in cost_block and "output" in cost_block:
                    # models.dev format
                    unit = cost_block.get("unit", "1K tokens")
                    input_price = cost_block.get("input", 0)
                    output_price = cost_block.get("output", 0)
                    report_lines.append(f"- **Input:** ${input_price} per {unit}")
                    report_lines.append(f"- **Output:** ${output_price} per {unit}")
                elif "input_per_million" in cost_block:
                    # app format
                    input_per_m = cost_block.get("input_per_million", 0)
                    output_per_m = cost_block.get("output_per_million", 0)
                    report_lines.append(f"- **Input:** ${input_per_m:.4f} per 1M tokens")
                    report_lines.append(f"- **Output:** ${output_per_m:.4f} per 1M tokens")
                
                report_lines.append("")
    
    # Processing Timeline
    with storage.get_db() as db:
        runs = (
            db.query(storage.Run)
            .filter(storage.Run.project_id == project_id)
            .order_by(storage.Run.created_at.asc())
            .all()
        )
        
        if runs:
            report_lines.append("## Processing Timeline")
            report_lines.append("")
            report_lines.append(f"- **First Run:** {runs[0].created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append(f"- **Last Run:** {runs[-1].created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append(f"- **Total Runs:** {len(runs)}")
            report_lines.append("")
    
    # Checkpoint Details (if available)
    if checkpoint_stats:
        report_lines.append("## Checkpoint Details")
        report_lines.append("")
        if checkpoint_dir:
            report_lines.append(f"- **Folder:** `{checkpoint_dir}`")
        
        # Get checkpoint metadata
        try:
            cp = FolderCheckpoint(checkpoint_dir) if checkpoint_dir else None
            if cp:
                cp.load()
                template_name = cp.data.get("template_name")
                model_id = cp.data.get("model_id")
                unstructured = cp.data.get("unstructured", False)
                
                if template_name:
                    report_lines.append(f"- **Template:** {template_name}")
                if model_id:
                    report_lines.append(f"- **Model:** {model_id}")
                report_lines.append(f"- **Mode:** {'Unstructured' if unstructured else 'Structured'}")
        except Exception:
            pass
        
        report_lines.append("")
    
    # Footer
    report_lines.append("---")
    report_lines.append("\n*Report generated by Images → JSON Project Tracker*")
    
    return "\n".join(report_lines)


def save_project_report(
    project_id: int,
    output_dir: Path,
    checkpoint_dir: Optional[Path] = None
) -> Path:
    """Generate and save project report to file.
    
    Returns:
        Path to saved report file
    """
    report_content = generate_project_report(project_id, checkpoint_dir)
    
    # Get project name for filename
    project = storage.get_project_by_id(project_id)
    project_name = project.name if project else f"project_{project_id}"
    
    # Sanitize filename
    safe_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in project_name)
    safe_name = safe_name.replace(' ', '_')
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"report_{safe_name}_{timestamp}.md"
    
    output_path = output_dir / filename
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_content, encoding="utf-8")
    
    return output_path


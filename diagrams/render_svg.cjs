// SVG layout adapter for canonical Mermaid flowcharts. This is a Graphviz
// structural export, not a Mermaid.js parser or renderer.
const fs = require('node:fs');
const path = require('node:path');

async function main() {
  const [specPath, outputPath, modulesPath] = process.argv.slice(2);
  if (!specPath || !outputPath || !modulesPath) {
    throw new Error('Usage: node render_svg.cjs SPEC.json OUT.svg NODE_MODULES');
  }
  const { instance } = require(path.join(modulesPath, '@viz-js/viz/dist/viz.cjs'));
  const spec = JSON.parse(fs.readFileSync(specPath, 'utf8'));
  const diagram = spec.diagrams[0];
  const quote = value => JSON.stringify(value);
  const classes = Object.fromEntries(diagram.styles.map(style => [style.name,
    Object.fromEntries(style.definition.split(',').map(item => item.split(':')))]));
  const dot = [
    'digraph G {',
    `graph [rankdir=${diagram.data.direction}, bgcolor="white", pad="0.3", nodesep="0.35", ranksep="0.55", splines=polyline, fontname="Segoe UI", fontsize=21, fontcolor="#0f172a", labelloc=t, label=${quote(spec.title)}];`,
    'node [shape=box, style="rounded,filled", fontname="Segoe UI", fontsize=15, margin="0.18,0.13", penwidth=1.3];',
    'edge [color="#64748b", penwidth=1.35, arrowsize=0.75, fontname="Segoe UI", fontsize=11, fontcolor="#475569"];',
  ];
  for (const node of diagram.data.nodes) {
    const style = classes[node.class] || {};
    dot.push(`${node.id} [label=${quote(node.label)}, fillcolor=${quote(style.fill || '#f8fafc')}, color=${quote(style.stroke || '#64748b')}, fontcolor=${quote(style.color || '#0f172a')}];`);
  }
  // Keep likelihood side branches subordinate to the principal prediction path.
  const ids = new Set(diagram.data.nodes.map(node => node.id));
  if (ids.has('marks') && ids.has('timing')) dot.push('{rank=same; marks; timing;}');
  const hasFeatures = diagram.data.nodes.some(node => node.id === 'feature_loss');
  if (ids.has('event_loss') && ids.has('censor_loss')) {
    dot.push(`{rank=same; event_loss; censor_loss; ${hasFeatures ? 'feature_loss;' : ''}}`);
  }
  if (ids.has('output') && ids.has('loss')) dot.push('{rank=same; output; loss;}');
  if (ids.has('latent_future') && ids.has('residual')) dot.push('{rank=same; latent_future; residual;}');
  for (const edge of diagram.data.edges) {
    const attrs = [];
    if (edge.label) attrs.push(`label=${quote(edge.label)}`);
    if (edge.arrow === '-.->') attrs.push('style=dashed');
    if (edge.arrow === '---') attrs.push('arrowhead=none');
    if (edge.to === 'feature_loss' || edge.to === 'loss') attrs.push('style=dashed', 'constraint=false', 'color="#e11d48"');
    dot.push(`${edge.from} -> ${edge.to} [${attrs.join(',')}];`);
  }
  dot.push('}');
  const viz = await instance();
  let svg = viz.renderString(dot.join('\n'), { format: 'svg', engine: 'dot' });
  svg = svg.replace(/<!DOCTYPE[\s\S]*?>\s*/, '');
  svg = svg.replace('<svg ', '<svg role="img" ');
  svg = svg.replace(/(<svg[^>]*>)/, '$1\n<desc>Structural SVG export of canonical Mermaid flowchart data; layout rendered by Graphviz via @viz-js/viz.</desc>');
  fs.writeFileSync(outputPath, svg);
  const sharp = require(path.join(modulesPath, 'sharp'));
  await sharp(Buffer.from(svg)).resize({ width: 1600 }).png().toFile(outputPath.replace(/\.svg$/, '.png'));
  fs.writeFileSync(outputPath.replace(/\.svg$/, '.dot'), dot.join('\n') + '\n');
  console.log(path.basename(outputPath) + ': Graphviz structural SVG export');
}

main().catch(error => { console.error(error.message); process.exitCode = 1; });

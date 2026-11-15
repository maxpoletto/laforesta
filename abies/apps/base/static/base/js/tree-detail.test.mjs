import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {Element, installDOM} from '../../../../../test/js-dom.mjs';

const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'abies-tree-detail-'));
const here = path.dirname(fileURLToPath(import.meta.url));
fs.cpSync(here, temporary, {recursive:true});
process.on('exit',()=>fs.rmSync(temporary,{recursive:true,force:true}));
fs.writeFileSync(path.join(temporary,'tree-points-map.js'), `
export class TreePointsMap {
  constructor(opts) { this.opts=opts; this.wrapper={syncBasemap:name=>this.basemap=name}; }
  setTrees(trees) { this.trees=trees; }
  destroy() { this.destroyed=true; }
  invalidateSize() {}
  fitParcels() {}
}
export const treePointsFromDigest=rows=>rows;
`);
fs.writeFileSync(path.join(temporary,'dendrometry.js'), `
export const aggregateTreeDendrometry=rows=>rows;
export const clearDendrometrySummaryInfo=()=>{};
export const renderDendrometryBarCharts=()=>({});
export const renderDendrometryLegend=()=>{};
export const renderDendrometrySummaryInfo=()=>{};
`);
const body=installDOM();
Element.prototype.before=function(node) {
  node.remove();
  const parent=this.parentNode;
  node.parentNode=parent;
  parent.children.splice(parent.children.indexOf(this),0,node);
};
const template=new Element('template');template.id='tmpl-tree-dendrometry-summary';
template.content=new Element('#fragment');
const root=new Element();root.className='tree-dendrometry-summary';
for(const target of ['status','chart-grid','species-row','species',
  'tree-count-info','volume-info','basal-area-info']) {
  const node=new Element();node.dataset.target=`dendrometry-${target}`;root.appendChild(node);
}
const exportButton=new Element('button');exportButton.dataset.action='export-dendrometry';
root.appendChild(exportButton);template.content.appendChild(root);body.appendChild(template);
globalThis.L={};
const {TreeDetail}=await import(pathToFileURL(path.join(temporary,'tree-detail.js')).href);
const digest={columns:[],rows:[[1],[2]]};
const geojson={type:'FeatureCollection',features:[]};

const defaultHost=new Element();
const original=new TreeDetail({container:defaultHost,digest,geojson});
assert.equal(defaultHost.children[0],original.mapHost,'Existing consumers retain map-before-summary order');
original.destroy();

const mapContainer=new Element(), summaryContainer=new Element();
const detail=new TreeDetail({container:summaryContainer,mapContainer,digest,geojson});
assert.equal(mapContainer.children[0],detail.mapHost,'Map can precede a separately owned table');
assert.equal(summaryContainer.children[0],detail.root,'Summary remains in its own host');
const map=detail.map;
detail.setData({...digest,rows:[]});
assert.equal(map.destroyed,true);
assert.equal(mapContainer.children.length,0,'Deleting the last tree removes the external map');
detail.syncBasemap('topo');
detail.setData(digest,{rows:[digest.rows[1]]});
assert.equal(detail.map.opts.basemap,'topo','A recreated map uses the latest basemap');
assert.deepEqual(detail.map.trees,[[2]],'A recreated map follows all retained rows');
detail.destroy();
assert.equal(mapContainer.children.length,0);
assert.equal(summaryContainer.children.length,0);

const structured=new TreeDetail({container:summaryContainer,digest,geojson,mapEnabled:false});
assert.equal(structured.map,null,'Area-map surveys can use the shared summary without a tree map');
structured.setData(digest,{rows:[digest.rows[0]]});
assert.equal(structured.map,null,'Source refresh must not enable a disabled tree map');
assert.deepEqual(structured.rows,[[1]]);
structured.destroy();
console.log('Tree detail: separate map/summary hosts, opt-out, filtering and empty-source lifecycle passed');

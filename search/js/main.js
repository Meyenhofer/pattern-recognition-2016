'use strict';

window.addEventListener('DOMContentLoaded', () => init(), false);

function init() {
  drawPathBBox();
  createSideBar();
  createTooltips();
}

function drawPathBBox() {
  let paths = document.getElementsByTagName('path');
  for (let i = 0; i < paths.length; i++) {
    let { x, y, width, height } = paths[i].getBBox();
    let pathElement = document.createElementNS('http://www.w3.org/2000/svg',
                                               'path');
    let pathText = `M ${x} ${y} L ${x + width} ${y}`;
    pathText += ` L ${x + width} ${y + height} L ${x} ${y + height} Z`;
    let pathId = paths[i].getAttribute('id');
    pathElement.setAttribute('d', pathText);
    pathElement.setAttribute('id', pathId);
    if (paths[i].classList.contains('highlight')) {
      pathElement.classList.add('highlight');
    }
    paths[i].parentElement.replaceChild(pathElement, paths[i]);
    pathElement.addEventListener('mouseenter', () => {
      let tooltip = document.getElementById(`tooltip-${pathId}`);
      showElement(tooltip.parentElement);
    });
    pathElement.addEventListener('mouseleave', () => {
      let tooltip = document.getElementById(`tooltip-${pathId}`);
      hideElement(tooltip.parentElement);
    });
  }
}

function createSideBar() {
  let div = document.createElement('div');
  div.setAttribute('id', 'sidebar');
  let span = document.createElement('span');
  span.textContent = document.title;
  span.setAttribute('id', 'sidebar-title');
  div.appendChild(span);
  addWords(div);
  document.body.appendChild(div);
}

function createTooltips() {
  let paths = document.getElementsByTagName('path');
  const tWidth = 110;
  const tHeight = 60;
  const center = tWidth / 2;
  const tipW = 10;
  const tipH = 15;
  for (let i = 0; i < paths.length; i++) {
    let pathId = paths[i].getAttribute('id');
    if (pathId.startsWith('tooltip')) {
      continue;
    }
    let { x, y, width, height } = paths[i].getBBox();
    let startX = x + (width - tWidth) / 2;
    let startY = y + height + 5;
    let gElement = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    let tip = `L ${center - tipW} 0 L ${center} ${-tipH} L ${center + tipW} 0`;
    let tooltip = createSVGTooltip(tWidth, tHeight, tip, pathId);
    gElement.setAttribute('transform', `translate(${startX}, ${startY})`);
    gElement.appendChild(tooltip);
    hideElement(gElement);
    let foreignObject = createTooltipForeignObject(tWidth, tHeight, pathId);
    gElement.appendChild(foreignObject);
    paths[i].parentElement.appendChild(gElement);
  }
}

function addWords(elem) {
  let paths = document.getElementsByTagName('path');
  for (let i = 0; i < paths.length; i++) {
    if (!paths[i].classList.contains('highlight')) {
      continue;
    }
    let div = document.createElement('div');
    let word = document.createElement('span');
    let visible = visibleIcon();
    let invisible = invisibleIcon();
    word.classList.add('word');
    word.textContent = paths[i].getAttribute('id');
    div.appendChild(visible);
    div.appendChild(invisible);
    div.appendChild(word);
    let path = document.getElementById(word.textContent);
    visible.addEventListener('click', () => {
      hideElement(visible);
      hideElement(path);
      showElement(invisible);
      word.classList.add('not-allowed');
    });
    invisible.addEventListener('click', () => {
      showElement(visible);
      showElement(path);
      hideElement(invisible);
      word.classList.remove('not-allowed');
    });
    word.addEventListener('click', () => {
      if (word.classList.contains('not-allowed')) {
        return;
      }
      let { left: x, top: y } = path.getBoundingClientRect();
      window.scrollBy(x - 200, y - 50);
    });
    elem.appendChild(div);
  }
}

function createSVGTooltip(width, height, tip, text) {
  let tooltip = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  let path = `M 0 0 ${tip} L ${width} 0 L ${width} ${height} L 0 ${height} Z`;
  tooltip.setAttribute('d', path);
  tooltip.setAttribute('class', 'tooltip');
  tooltip.setAttribute('id', `tooltip-${text}`);

  return tooltip;
}

function createTooltipForeignObject(width, height, text) {
  let tooltipText = document.createElementNS('http://www.w3.org/2000/svg',
                                              'foreignObject');
  tooltipText.setAttribute('width', width);
  tooltipText.setAttribute('height', height);
  let div = document.createElement('div');
  div.classList.add('tooltip-div');
  let span = document.createElement('span');
  span.textContent = text;
  span.classList.add('tooltip-text');
  div.appendChild(span);
  tooltipText.appendChild(div);

  return tooltipText;
}

function showElement(elem) {
  elem.classList.remove('invisible');
}

function hideElement(elem) {
  elem.classList.add('invisible');
}

function visibleIcon() {
  let image = document.createElement('img');
  image.setAttribute('src', 'search/assets/visibility.png');
  image.setAttribute('class', 'checkicon');

  return image;
}

function invisibleIcon() {
  let image = document.createElement('img');
  image.setAttribute('src', 'search/assets/visibility_off.png');
  image.setAttribute('class', 'checkicon invisible');

  return image;
}

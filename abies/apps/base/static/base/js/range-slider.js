/**
 * Dual-thumb range slider (ES module).
 * Ported from bosco/a/range-slider.js.
 *
 * @param {HTMLInputElement} minInput
 * @param {HTMLInputElement} maxInput
 * @param {HTMLElement} labelEl — displays current range as text
 * @param {function} onChange — called after every thumb move
 * @returns {{ setRange(values: number[]), setValues(lo: number, hi: number), getRange(): [number, number] }}
 */
export function createRangeSlider(minInput, maxInput, labelEl, onChange) {
  function updateLabel() {
    const a = minInput.value, b = maxInput.value;
    labelEl.textContent = a === b ? a : a + '\u2013' + b;
  }

  function onInput() {
    const lo = parseInt(minInput.value, 10);
    const hi = parseInt(maxInput.value, 10);
    if (lo > hi) {
      if (this === minInput) minInput.value = maxInput.value;
      else maxInput.value = minInput.value;
    }
    updateLabel();
    onChange();
  }

  minInput.addEventListener('input', onInput);
  maxInput.addEventListener('input', onInput);

  return {
    /** Set min/max bounds and reset thumbs to endpoints. */
    setRange(values) {
      const lo = values[0], hi = values[values.length - 1];
      minInput.min = maxInput.min = lo;
      minInput.max = maxInput.max = hi;
      minInput.value = lo;
      maxInput.value = hi;
      updateLabel();
    },
    /** Set thumb positions without changing min/max bounds. */
    setValues(lo, hi) {
      minInput.value = lo;
      maxInput.value = hi;
      updateLabel();
    },
    /** Return current [lo, hi] as integers. */
    getRange() {
      return [parseInt(minInput.value, 10), parseInt(maxInput.value, 10)];
    },
  };
}

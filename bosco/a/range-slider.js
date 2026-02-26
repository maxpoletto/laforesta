// Shared dual-thumb range slider module.
// Usage: createRangeSlider(minInput, maxInput, labelEl, onChange)

/**
 * Attach event listeners to a pair of range inputs that form a dual-thumb slider.
 * @param {HTMLInputElement} minInput - The "low" range input.
 * @param {HTMLInputElement} maxInput - The "high" range input.
 * @param {HTMLElement} labelEl - Element to display the current range as text.
 * @param {function} onChange - Callback invoked after every thumb move.
 * @returns {{ setRange(values: number[]), getRange(): [number, number] }}
 */
function createRangeSlider(minInput, maxInput, labelEl, onChange) {
    function updateLabel() {
        var a = minInput.value, b = maxInput.value;
        labelEl.textContent = a === b ? a : a + '\u2013' + b;
    }

    function onInput() {
        var lo = parseInt(minInput.value, 10);
        var hi = parseInt(maxInput.value, 10);
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
        /** Set min/max/value from an array of sorted values. */
        setRange: function(values) {
            var lo = values[0], hi = values[values.length - 1];
            minInput.min = maxInput.min = lo;
            minInput.max = maxInput.max = hi;
            minInput.value = lo;
            maxInput.value = hi;
            updateLabel();
        },
        /** Return current [lo, hi] as integers. */
        getRange: function() {
            return [parseInt(minInput.value, 10), parseInt(maxInput.value, 10)];
        },
    };
}

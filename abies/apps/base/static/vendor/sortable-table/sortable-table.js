/**
 * SortableTable - A reusable, configurable sortable table with pagination
 */
(function() {
    'use strict';

class SortableTable {
    constructor(options = {}) {
        // Validate required options
        if (!options.container) {
            throw new Error('Container element or selector is required');
        }
        if (!options.data || !Array.isArray(options.data)) {
            throw new Error('Data array is required');
        }
        if (!options.columns || !Array.isArray(options.columns)) {
            throw new Error('Columns array is required');
        }

        // Set up configuration with defaults
        this.container = typeof options.container === 'string'
            ? document.querySelector(options.container)
            : options.container;

        if (!this.container) {
            throw new Error(`Container element not found (${options.container})`);
        }

        this.data = [...options.data]; // Create a copy to avoid mutating original
        this.originalData = [...options.data]; // Keep original for filtering
        this.columns = [...options.columns];
        this.rowsPerPage = options.rowsPerPage || 25;
        this.showPagination = options.showPagination !== false;
        this.allowSorting = options.allowSorting !== false;
        this.cssPrefix = options.cssPrefix || 'sortable-table';
        this.emptyMessage = options.emptyMessage || 'No data available';

        // Sorting state

        if (options.sort && options.sort.column) {
            this.currentSort = { column: options.sort.column, ascending: options.sort.ascending ?? true };
        } else {
            this.currentSort = { column: null, ascending: true };
        }

        // Filter state
        this.currentFilter = null;
        if (this.currentSort.column) {
            this._sortData(this.columns.findIndex(col => col.key === this.currentSort.column),
                this.columns.find(col => col.key === this.currentSort.column).type,
                this.currentSort.ascending);
        }

        // Pagination state
        this.currentPage = 1;
        this.totalPages = Math.max(1, Math.ceil(this.data.length / this.rowsPerPage));

        // Event callbacks
        this.onSort = options.onSort || null;
        this.onPageChange = options.onPageChange || null;
        this.onRowClick = options.onRowClick || null;

        // Initialize
        this.init();
    }

    init() {
        // Append the container class if not already present
        const containerClass = `${this.cssPrefix}-container`;
        if (!this.container.classList.contains(containerClass)) {
            this.container.classList.add(containerClass);
        }
        this.render();
    }

    render() {
        this.container.innerHTML = this.generateHTML();
        this.updatePaginationState();
        this.updateSortIndicators();
        this.attachEventListeners();
    }

    generateHTML() {
        return `
            ${this.showPagination ? this.generatePaginationHTML() : ''}
            <div class="${this.cssPrefix}-wrapper">
                <table class="${this.cssPrefix}">
                    <thead class="${this.cssPrefix}-head">
                        <tr>
                            ${this.generateHeaderHTML()}
                        </tr>
                    </thead>
                    <tbody class="${this.cssPrefix}-body">
                        ${this.generateBodyHTML()}
                    </tbody>
                </table>
            </div>
        `;
    }

    generatePaginationHTML() {
        return `
            <div class="${this.cssPrefix}-controls">
                <div class="${this.cssPrefix}-pagination">
                    <button class="first-page" data-action="first">&lt;&lt;</button>
                    <button class="prev-page" data-action="prev">&lt;</button>
                    <span class="page-info">
                        Page <span class="current-page-number">${this.currentPage}</span>
                        of <span class="total-pages">${this.totalPages}</span>
                    </span>
                    <button class="next-page" data-action="next">&gt;</button>
                    <button class="last-page" data-action="last">&gt;&gt;</button>
                </div>
            </div>
        `;
    }

    generateHeaderHTML() {
        return this.columns.map((col, index) => {
            if (col.hidden) return '';
            const sortable = this.allowSorting && col.sortable !== false;
            const classes = [
                `${this.cssPrefix}-header`,
                sortable ? 'sortable' : '',
                (col.type === 'number' || col.type === 'date') ? 'numeric' : '',
                col.className || ''
            ].filter(Boolean).join(' ');

            return `
                <th class="${classes}"
                    data-column="${col.key}"
                    data-index="${index}"
                    data-type="${col.type || 'string'}"
                    ${col.width ? `style="width: ${col.width}"` : ''}>
                    ${col.label}
                    ${sortable ? `<span class="sort-indicator">↕</span>` : ''}
                </th>
            `;
        }).join('');
    }

    generateBodyHTML() {
        if (this.data.length === 0) {
            return `
                <tr>
                    <td colspan="${this.columns.filter(col => !col.hidden).length}" class="${this.cssPrefix}-empty">
                        ${this.emptyMessage}
                    </td>
                </tr>
            `;
        }

        const startIndex = (this.currentPage - 1) * this.rowsPerPage;
        const endIndex = Math.min(startIndex + this.rowsPerPage, this.data.length);
        const pageData = this.data.slice(startIndex, endIndex);

        return pageData.map((row, rowIndex) => {
            const actualIndex = startIndex + rowIndex;
            return `
                <tr class="${this.cssPrefix}-row" data-index="${actualIndex}">
                    ${this.columns.map((col, colIndex) => {
                        if (col.hidden) return '';
                        const value = row[colIndex] ?? '';
                        const formatted = this.formatCellValue(value, col);
                        const classes = [
                            `${this.cssPrefix}-cell`,
                            (col.type === 'number' || col.type === 'date') ? 'numeric' : '',
                            col.cellClassName || ''
                        ].filter(Boolean).join(' ');

                        return `<td class="${classes}" data-column="${col.key}">${formatted}</td>`;
                    }).join('')}
                </tr>
            `;
        }).join('');
    }

    formatCellValue(value, column) {
        if (column.formatter && typeof column.formatter === 'function') {
            return column.formatter(value);
        }

        switch (column.type) {
            case 'number':
                return typeof value === 'number' ? value.toLocaleString() : value;
            case 'date':
                return value instanceof Date ? value.toLocaleDateString() : value;
            case 'boolean':
                return value ? 'Yes' : 'No';
            default:
                return String(value);
        }
    }

    attachEventListeners() {
        // Pagination controls
        if (this.showPagination) {
            this.attachPaginationListeners();
        }

        // Column sorting
        if (this.allowSorting) {
            this.attachSortingListeners();
        }

        // Row clicks
        if (this.onRowClick) {
            this.attachRowClickListeners();
        }
    }

    attachPaginationListeners() {
        const pagination = this.container.querySelector(`.${this.cssPrefix}-pagination`);
        if (!pagination) return;

        // Navigation buttons
        pagination.addEventListener('click', (e) => {
            if (e.target.matches('button[data-action]')) {
                const action = e.target.dataset.action;
                this.handlePaginationAction(action);
            }
        });

        // Page number click to edit
        const pageNumber = pagination.querySelector('.current-page-number');
        if (pageNumber) {
            pageNumber.addEventListener('click', (e) => {
                this.makePageNumberEditable(e.target);
            });
        }
    }

    makePageNumberEditable(element) {
        const input = document.createElement('input');
        input.type = 'number';
        input.min = '1';
        input.max = this.totalPages.toString();
        input.value = this.currentPage.toString();
        input.className = 'page-number-input';
        input.style.width = '50px';
        input.style.textAlign = 'center';

        const finishEdit = () => {
            const newPage = parseInt(input.value);
            if (newPage >= 1 && newPage <= this.totalPages && newPage !== this.currentPage) {
                this.goToPage(newPage);
            }
            // Restore the span
            const newSpan = document.createElement('span');
            newSpan.className = 'current-page-number';
            newSpan.textContent = this.currentPage.toString();
            newSpan.addEventListener('click', (e) => this.makePageNumberEditable(e.target));
            input.parentNode.replaceChild(newSpan, input);
        };

        input.addEventListener('blur', finishEdit);
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                finishEdit();
            } else if (e.key === 'Escape') {
                // Cancel edit
                const newSpan = document.createElement('span');
                newSpan.className = 'current-page-number';
                newSpan.textContent = this.currentPage.toString();
                newSpan.addEventListener('click', (e) => this.makePageNumberEditable(e.target));
                input.parentNode.replaceChild(newSpan, input);
            }
        });

        element.parentNode.replaceChild(input, element);
        input.focus();
        input.select();
    }

    attachSortingListeners() {
        const headers = this.container.querySelectorAll('th.sortable');
        headers.forEach(header => {
            header.addEventListener('click', () => {
                const column = header.dataset.column;
                const type = header.dataset.type;
                this.toggleSort(column, type);
            });
        });
    }

    attachRowClickListeners() {
        const tbody = this.container.querySelector(`.${this.cssPrefix}-body`);
        tbody.addEventListener('click', (e) => {
            const row = e.target.closest(`.${this.cssPrefix}-row`);
            if (row) {
                const index = parseInt(row.dataset.index);
                // Get the row data from the filtered/sorted data array, not originalData
                const data = this.data[index];
                this.onRowClick(data, index, e);
            }
        });
    }

    handlePaginationAction(action) {
        switch (action) {
            case 'first':
                this.goToPage(1);
                break;
            case 'prev':
                if (this.currentPage > 1) {
                    this.goToPage(this.currentPage - 1);
                }
                break;
            case 'next':
                if (this.currentPage < this.totalPages) {
                    this.goToPage(this.currentPage + 1);
                }
                break;
            case 'last':
                this.goToPage(this.totalPages);
                break;
        }
    }

    goToPage(page) {
        if (page >= 1 && page <= this.totalPages && page !== this.currentPage) {
            this.currentPage = page;
            this.updateTable();

            if (this.onPageChange) {
                this.onPageChange(page, this.totalPages);
            }
        }
    }

    _sortData(columnIndex, columnType, ascending) {
        // Sort both originalData and data to maintain consistency
        const sortFn = (a, b) => {
            const aVal = a[columnIndex] ?? '';
            const bVal = b[columnIndex] ?? '';

            let comparison = 0;

            switch (columnType) {
                case 'number':
                    const aNum = parseFloat(aVal) || 0;
                    const bNum = parseFloat(bVal) || 0;
                    comparison = aNum - bNum;
                    break;
                case 'date':
                    const aDate = new Date(aVal || '1900-01-01');
                    const bDate = new Date(bVal || '1900-01-01');
                    comparison = aDate - bDate;
                    break;
                default:
                    comparison = String(aVal).toLowerCase().localeCompare(String(bVal).toLowerCase());
            }

            return ascending ? comparison : -comparison;
        };

        // Sort the source data
        this.originalData.sort(sortFn);

        // Re-apply current filter to get sorted, filtered data
        if (this.currentFilter) {
            this.data = this.originalData.filter(this.currentFilter);
        } else {
            this.data = [...this.originalData];
        }
    }

    toggleSort(columnKey, columnType) {
        const isCurrentColumn = this.currentSort.column === columnKey;
        const newOrder = isCurrentColumn ? !this.currentSort.ascending : true;

        this.sort(columnKey, columnType, newOrder);
    }

    sort(columnKey, columnType, ascending = true) {
        this.currentSort = { column: columnKey, ascending: ascending };
        const columnIndex = this.columns.findIndex(col => col.key === columnKey);
        this._sortData(columnIndex, columnType, ascending);
        this.currentPage = 1;
        this.updateTable();

        if (this.onSort) {
            this.onSort(columnKey, ascending);
        }
    }

    updateTable() {
        const tbody = this.container.querySelector(`.${this.cssPrefix}-body`);
        tbody.innerHTML = this.generateBodyHTML();
        this.updatePaginationState();
        this.updateSortIndicators();
        // Re-attach row click listeners for new rows
        if (this.onRowClick) {
            this.attachRowClickListeners();
        }
    }

    updatePaginationState() {
        if (!this.showPagination) return;

        const pagination = this.container.querySelector(`.${this.cssPrefix}-pagination`);
        if (!pagination) return;

        // Update page numbers
        const currentPageSpan = pagination.querySelector('.current-page-number');
        const totalPagesSpan = pagination.querySelector('.total-pages');

        if (currentPageSpan) currentPageSpan.textContent = this.currentPage;
        if (totalPagesSpan) totalPagesSpan.textContent = this.totalPages;

        // Update button states
        const firstBtn = pagination.querySelector('.first-page');
        const prevBtn = pagination.querySelector('.prev-page');
        const nextBtn = pagination.querySelector('.next-page');
        const lastBtn = pagination.querySelector('.last-page');

        if (firstBtn) firstBtn.disabled = this.currentPage <= 1;
        if (prevBtn) prevBtn.disabled = this.currentPage <= 1;
        if (nextBtn) nextBtn.disabled = this.currentPage >= this.totalPages;
        if (lastBtn) lastBtn.disabled = this.currentPage >= this.totalPages;
    }

    updateSortIndicators() {
        if (!this.allowSorting) return;

        // Clear all indicators
        const indicators = this.container.querySelectorAll('.sort-indicator');
        indicators.forEach(indicator => {
            indicator.className = 'sort-indicator';
            indicator.textContent = '↕';
        });

        // Set current sort indicator
        if (this.currentSort.column) {
            const currentHeader = this.container.querySelector(`th[data-column="${this.currentSort.column}"] .sort-indicator`);
            if (currentHeader) {
                currentHeader.className = `sort-indicator ${this.currentSort.ascending ? 'asc' : 'desc'}`;
                currentHeader.textContent = this.currentSort.ascending ? '↑' : '↓';
            }
        }
    }

    // Public API methods
    setData(newData) {
        this.data = [...newData];
        this.originalData = [...newData];
        this.totalPages = Math.ceil(this.data.length / this.rowsPerPage);
        this.currentPage = 1;
        this.updateTable();
    }

    addRow(rowData) {
        this.originalData.push(rowData);

        // Only add to filtered data if it passes the current filter (or no filter is active)
        if (!this.currentFilter || this.currentFilter(rowData)) {
            this.data.push(rowData);
        }

        this.totalPages = Math.ceil(this.data.length / this.rowsPerPage);
        this.updateTable();
    }

    removeRows(predicate) {
        // Remove from original data
        const originalLength = this.originalData.length;
        this.originalData = this.originalData.filter(row => !predicate(row));
        const removedCount = originalLength - this.originalData.length;

        // Remove from filtered/sorted data
        this.data = this.data.filter(row => !predicate(row));

        this.totalPages = Math.ceil(this.data.length / this.rowsPerPage);

        if (this.currentPage > this.totalPages) {
            this.currentPage = Math.max(1, this.totalPages);
        }

        this.updateTable();
        return removedCount;
    }

    filter(predicate) {
        this.currentFilter = predicate;
        this.data = this.originalData.filter(predicate);
        this.totalPages = Math.ceil(this.data.length / this.rowsPerPage);
        this.currentPage = 1;
        this.updateTable();
    }

    clearFilter() {
        this.currentFilter = null;
        this.data = [...this.originalData];
        this.totalPages = Math.ceil(this.data.length / this.rowsPerPage);
        this.currentPage = 1;
        this.updateTable();
    }

    getVisibleData() {
        const startIndex = (this.currentPage - 1) * this.rowsPerPage;
        const endIndex = Math.min(startIndex + this.rowsPerPage, this.data.length);
        return this.data.slice(startIndex, endIndex);
    }

    destroy() {
        this.container.innerHTML = '';
        this.container.classList.remove(this.cssPrefix + '-container');
        this.data = this.originalData = this.columns = this.currentFilter = null;
    }
}

// Export for use in other files
// Ensure SortableTable is available globally
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SortableTable;
}

// Always attach to window for browser compatibility
if (typeof window !== 'undefined') {
    window.SortableTable = SortableTable;
}

// Additional Safari compatibility check
if (typeof window !== 'undefined' && typeof window.SortableTable === 'undefined') {
    console.error('Failed to attach SortableTable to window object');
}

})(); // Close the IIFE

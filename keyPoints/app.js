// 知识点数据结构
const knowledgeData = [
    {
        id: 'SE001',
        title: '软件生命周期模型',
        category: 'SE',
        categoryName: '软件工程基础',
        importance: '高',
        types: ['选择题', '案例题'],
        mustLearn: true,
        summary: '软件生命周期模型是指导软件开发全过程的框架，瀑布模型适合需求稳定项目，增量模型适合需求变化项目，原型模型适合需求不明确项目。',
        keywords: ['瀑布模型', '增量模型', '螺旋模型', '原型模型', '敏捷模型', '生命周期']
    },
    {
        id: 'SE002',
        title: '需求工程',
        category: 'SE',
        categoryName: '软件工程基础',
        importance: '高',
        types: ['选择题', '案例题'],
        mustLearn: true,
        summary: '需求工程包括需求获取、需求分析、需求规约、需求验证四个阶段，功能需求描述系统做什么，非功能需求描述系统如何做，需求跟踪确保需求变更的可控性。',
        keywords: ['需求获取', '需求分析', '功能需求', '非功能需求', 'SRS', '需求验证']
    },
    {
        id: 'OO001',
        title: '面向对象基本概念',
        category: 'OO',
        categoryName: '面向对象技术',
        importance: '高',
        types: ['选择题', '案例题'],
        mustLearn: true,
        summary: '面向对象三大特征是封装、继承、多态，四大基本概念是对象、类、封装、继承，核心思想是将数据和操作数据的方法绑定在一起。',
        keywords: ['封装', '继承', '多态', '对象', '类', '抽象']
    },
    {
        id: 'OO002',
        title: 'UML用例图',
        category: 'OO',
        categoryName: '面向对象技术',
        importance: '高',
        types: ['选择题', '案例题'],
        mustLearn: true,
        summary: '用例图描述系统功能需求，包含参与者、用例、关系三要素，关系有关联、包含、扩展、泛化四种，用于需求分析阶段。',
        keywords: ['用例图', '参与者', '用例', 'include', 'extend', 'UML']
    },
    {
        id: 'DB001',
        title: '关系数据库理论',
        category: 'DB',
        categoryName: '数据库技术',
        importance: '高',
        types: ['选择题', '案例题'],
        mustLearn: true,
        summary: '关系数据库基于关系模型，数据存储在二维表中，通过主键唯一标识记录，外键建立表间联系，满足实体完整性、参照完整性、用户定义完整性三大约束。',
        keywords: ['关系模型', '主键', '外键', '完整性约束', '函数依赖', '关系代数']
    },
    {
        id: 'DS001',
        title: '线性数据结构',
        category: 'DS',
        categoryName: '数据结构算法',
        importance: '中',
        types: ['选择题'],
        mustLearn: false,
        summary: '线性数据结构包括数组、链表、栈、队列，特点是元素间存在一对一的线性关系，栈是后进先出（LIFO），队列是先进先出（FIFO）。',
        keywords: ['数组', '链表', '栈', '队列', 'LIFO', 'FIFO']
    }
];

// 应用状态
let currentView = 'list';
let filteredData = [...knowledgeData];

// DOM 元素
const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const categoryFilter = document.getElementById('categoryFilter');
const importanceFilter = document.getElementById('importanceFilter');
const typeFilter = document.getElementById('typeFilter');
const knowledgeList = document.getElementById('knowledgeList');
const listViewBtn = document.getElementById('listView');
const cardViewBtn = document.getElementById('cardView');
const detailModal = document.getElementById('detailModal');
const detailContent = document.getElementById('detailContent');
const totalCount = document.getElementById('totalCount');
const highCount = document.getElementById('highCount');
const mustLearnCount = document.getElementById('mustLearnCount');

// 初始化
document.addEventListener('DOMContentLoaded', function() {
    updateStats();
    renderKnowledgeList();
    bindEvents();
});

// 绑定事件
function bindEvents() {
    // 搜索功能
    searchBtn.addEventListener('click', handleSearch);
    searchInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            handleSearch();
        }
    });
    
    // 筛选功能
    categoryFilter.addEventListener('change', handleFilter);
    importanceFilter.addEventListener('change', handleFilter);
    typeFilter.addEventListener('change', handleFilter);
    
    // 视图切换
    listViewBtn.addEventListener('click', () => switchView('list'));
    cardViewBtn.addEventListener('click', () => switchView('card'));
    
    // 模态框关闭
    document.querySelector('.close').addEventListener('click', closeModal);
    window.addEventListener('click', function(e) {
        if (e.target === detailModal) {
            closeModal();
        }
    });
    
    // 快速导航
    document.querySelectorAll('.nav-category').forEach(category => {
        category.addEventListener('click', function() {
            const categoryCode = this.dataset.category;
            categoryFilter.value = categoryCode;
            handleFilter();
        });
    });
}

// 搜索处理
function handleSearch() {
    const query = searchInput.value.toLowerCase().trim();
    if (!query) {
        filteredData = [...knowledgeData];
    } else {
        filteredData = knowledgeData.filter(item => {
            return item.title.toLowerCase().includes(query) ||
                   item.summary.toLowerCase().includes(query) ||
                   item.keywords.some(keyword => keyword.toLowerCase().includes(query)) ||
                   item.categoryName.toLowerCase().includes(query);
        });
    }
    applyFilters();
    renderKnowledgeList();
}

// 筛选处理
function handleFilter() {
    applyFilters();
    renderKnowledgeList();
}

// 应用筛选条件
function applyFilters() {
    let data = [...knowledgeData];
    
    // 搜索筛选
    const query = searchInput.value.toLowerCase().trim();
    if (query) {
        data = data.filter(item => {
            return item.title.toLowerCase().includes(query) ||
                   item.summary.toLowerCase().includes(query) ||
                   item.keywords.some(keyword => keyword.toLowerCase().includes(query)) ||
                   item.categoryName.toLowerCase().includes(query);
        });
    }
    
    // 分类筛选
    const category = categoryFilter.value;
    if (category) {
        data = data.filter(item => item.category === category);
    }
    
    // 重要性筛选
    const importance = importanceFilter.value;
    if (importance) {
        data = data.filter(item => item.importance === importance);
    }
    
    // 题型筛选
    const type = typeFilter.value;
    if (type) {
        data = data.filter(item => item.types.includes(type));
    }
    
    filteredData = data;
}

// 渲染知识点列表
function renderKnowledgeList() {
    if (filteredData.length === 0) {
        knowledgeList.innerHTML = '<div style="text-align: center; padding: 40px; color: #666;">暂无匹配的知识点</div>';
        return;
    }
    
    knowledgeList.className = `knowledge-list ${currentView === 'card' ? 'card-view' : ''}`;
    
    knowledgeList.innerHTML = filteredData.map(item => `
        <div class="knowledge-item ${currentView === 'card' ? 'card-view' : ''}" onclick="showDetail('${item.id}')">
            <div class="knowledge-header">
                <div>
                    <div class="knowledge-title">${item.id} - ${item.title}</div>
                    <div class="knowledge-tags">
                        <span class="tag category">${item.categoryName}</span>
                        <span class="tag importance-${item.importance === '高' ? 'high' : item.importance === '中' ? 'medium' : 'low'}">
                            ${item.importance}频
                        </span>
                        ${item.mustLearn ? '<span class="tag must-learn">必背</span>' : ''}
                        ${item.types.map(type => `<span class="tag">${type}</span>`).join('')}
                    </div>
                </div>
            </div>
            <div class="knowledge-summary">${item.summary}</div>
        </div>
    `).join('');
}

// 切换视图
function switchView(view) {
    currentView = view;
    listViewBtn.classList.toggle('active', view === 'list');
    cardViewBtn.classList.toggle('active', view === 'card');
    renderKnowledgeList();
}

// 显示详情
async function showDetail(id) {
    try {
        const response = await fetch(`knowledge-base/${id}-${getKnowledgeTitle(id)}.md`);
        if (!response.ok) {
            throw new Error('文件不存在');
        }
        
        const content = await response.text();
        const htmlContent = parseMarkdownToHTML(content);
        
        detailContent.innerHTML = `<div class="detail-content">${htmlContent}</div>`;
        detailModal.style.display = 'block';
    } catch (error) {
        detailContent.innerHTML = `
            <div class="detail-content">
                <h1>知识点详情</h1>
                <p style="color: #666; text-align: center; padding: 40px;">
                    暂时无法加载详细内容，请稍后再试。
                </p>
            </div>
        `;
        detailModal.style.display = 'block';
    }
}

// 获取知识点标题
function getKnowledgeTitle(id) {
    const item = knowledgeData.find(item => item.id === id);
    return item ? item.title : '';
}

// 简单的 Markdown 转 HTML
function parseMarkdownToHTML(markdown) {
    return markdown
        .replace(/^# (.*$)/gm, '<h1>$1</h1>')
        .replace(/^## (.*$)/gm, '<h2>$1</h2>')
        .replace(/^### (.*$)/gm, '<h3>$1</h3>')
        .replace(/^\> (.*$)/gm, '<blockquote>$1</blockquote>')
        .replace(/^\- (.*$)/gm, '<li>$1</li>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/【题目】(.*?)【正确答案】(.*?)【解析】(.*?)(?=\n\n|\n$|$)/gs, 
            '<div class="example"><strong>【题目】</strong>$1<br><strong>【正确答案】</strong>$2<br><strong>【解析】</strong>$3</div>')
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>')
        .replace(/^/, '<p>')
        .replace(/$/, '</p>')
        .replace(/<p><\/p>/g, '')
        .replace(/<p>(<h[1-6]>)/g, '$1')
        .replace(/(<\/h[1-6]>)<\/p>/g, '$1')
        .replace(/<p>(<li>.*?<\/li>)<\/p>/g, '<ul>$1</ul>')
        .replace(/<br><li>/g, '<li>')
        .replace(/<\/li><br>/g, '</li>');
}

// 关闭模态框
function closeModal() {
    detailModal.style.display = 'none';
}

// 更新统计信息
function updateStats() {
    totalCount.textContent = knowledgeData.length;
    highCount.textContent = knowledgeData.filter(item => item.importance === '高').length;
    mustLearnCount.textContent = knowledgeData.filter(item => item.mustLearn).length;
}

// 实时搜索
searchInput.addEventListener('input', function() {
    clearTimeout(this.searchTimeout);
    this.searchTimeout = setTimeout(() => {
        handleSearch();
    }, 300);
});
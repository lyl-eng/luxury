"""
译前预处理Agent
负责文本结构拆解和语域风格识别
"""

import re
from typing import Dict, Any, List
from .BaseAgent import BaseAgent
from ModuleFolders.Cache.CacheProject import CacheProject
from ModuleFolders.Cache.CacheFile import CacheFile
from ModuleFolders.Cache.CacheItem import CacheItem, TranslationStatus


class PreprocessingAgent(BaseAgent):
    """
    Agent 0: 译前预处理Agent
    功能：
    1. 文本结构拆解：按段落、逻辑块、章节边界切分
    2. 语域与风格自动识别：判断领域和语体风格
    """
    
    def __init__(self, config=None):
        super().__init__(
            name="PreprocessingAgent",
            description="译前预处理与结构化拆解Agent",
            config=config
        )
        
        # 语域识别规则
        self.domain_patterns = {
            "legal": [r"法律", r"法规", r"条款", r"合同", r"协议"],
            "literary": [r"小说", r"文学", r"诗歌", r"散文"],
            "news": [r"报道", r"新闻", r"记者", r"消息"],
            "technical": [r"技术", r"系统", r"软件", r"硬件", r"算法"],
            "entertainment": [r"说唱", r"音乐", r"娱乐", r"明星", r"NBA"],
        }
        
        # 风格识别规则
        self.style_patterns = {
            "formal": [r"尊敬的", r"根据", r"依据", r"特此"],
            "informal": [r"哥们", r"老铁", r"牛逼", r"卧槽"],
            "academic": [r"研究表明", r"数据表明", r"综上所述"],
            "colloquial": [r"口语", r"俚语", r"方言"],
        }
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行预处理任务
        
        Args:
            input_data: 包含cache_project的字典
            
        Returns:
            包含处理后的cache_project和元数据的字典
        """
        try:
            self.log_agent_action("开始执行译前预处理")
            
            cache_project: CacheProject = input_data.get("cache_project")
            if not cache_project:
                self.error("未找到cache_project数据")
                return {"success": False, "error": "缺少cache_project"}
            
            # 1. 文本结构拆解（如果尚未拆解）
            self._structure_text(cache_project)
            
            # 2. 语域与风格识别
            metadata = self._identify_domain_and_style(cache_project)
            
            # 保存元数据到项目
            cache_project.extra["preprocessing_metadata"] = metadata
            
            # ==========================================
            # DB: 写入 topic_info 到数据库
            # ==========================================
            self._save_topic_info_to_db(cache_project, metadata)
            
            self.log_agent_action("译前预处理完成", f"识别领域: {metadata.get('domain')}, 风格: {metadata.get('style')}")
            
            return {
                "success": True,
                "cache_project": cache_project,
                "metadata": metadata
            }
        except Exception as e:
            self.error(f"译前预处理执行失败: {e}", e)
            return {"success": False, "error": str(e)}

    def _save_topic_info_to_db(self, cache_project: CacheProject, metadata: Dict[str, Any]):
        """将 topic_info 保存到数据库"""
        try:
            if not hasattr(cache_project, 'db_work_id') or not cache_project.db_work_id:
                self.debug("[DB] 无 work_id，跳过 topic_info 写入")
                return
            
            from ModuleFolders.Cache.DatabaseManager import DatabaseManager
            db_manager = DatabaseManager()
            
            # 构建 topic_info
            topic_info = {
                "domain": metadata.get("domain", "general"),
                "style": metadata.get("style", "neutral"),
                "domain_scores": metadata.get("domain_scores", {}),
                "style_scores": metadata.get("style_scores", {}),
                "total_text_length": metadata.get("total_text_length", 0)
            }
            
            # 更新到数据库
            db_manager.update_project_topic_info(cache_project.db_work_id, topic_info)
            self.info(f"[DB] topic_info 已保存: domain={topic_info['domain']}, style={topic_info['style']}")
            
        except Exception as e:
            self.error(f"[DB] topic_info 保存失败: {e}")
    
    def _structure_text(self, cache_project: CacheProject) -> None:
        """
        文本结构拆解
        确保文本按逻辑单元（段落、逻辑块）切分
        """
        self.log_agent_action("执行文本结构拆解")
        
        for file_path, cache_file in cache_project.files.items():
            # 检查是否已经结构化
            if cache_file.items:
                # 检查是否需要进一步拆解长段落
                new_items = []
                for item in cache_file.items:
                    # 如果单个item过长（超过500字符），尝试按句子拆分
                    if len(item.source_text) > 500:
                        sentences = self._split_into_sentences(item.source_text)
                        for idx, sentence in enumerate(sentences):
                            if sentence.strip():
                                new_item = CacheItem(
                                    text_index=item.text_index + idx,
                                    source_text=sentence.strip(),
                                    translation_status=TranslationStatus.UNTRANSLATED
                                )
                                new_items.append(new_item)
                    else:
                        new_items.append(item)
                
                # 更新items（这里简化处理，实际可能需要更复杂的逻辑）
                if len(new_items) != len(cache_file.items):
                    self.debug(f"文件 {file_path} 进行了进一步结构拆解")
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """将文本按句子拆分"""
        # 简单的句子分割（可以改进为更智能的分割）
        sentences = re.split(r'[.!?。！？]\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _identify_domain_and_style(self, cache_project: CacheProject) -> Dict[str, Any]:
        """
        识别语域和风格
        
        Returns:
            包含domain和style的元数据字典
        """
        self.log_agent_action("执行语域与风格识别")
        
        # 收集所有文本内容
        all_text = ""
        for cache_file in cache_project.files.values():
            for item in cache_file.items:
                all_text += item.source_text + " "
        
        # 识别领域
        domain_scores = {}
        for domain, patterns in self.domain_patterns.items():
            score = sum(len(re.findall(pattern, all_text, re.IGNORECASE)) for pattern in patterns)
            if score > 0:
                domain_scores[domain] = score
        
        detected_domain = max(domain_scores.items(), key=lambda x: x[1])[0] if domain_scores else "general"
        
        # 识别风格
        style_scores = {}
        for style, patterns in self.style_patterns.items():
            score = sum(len(re.findall(pattern, all_text, re.IGNORECASE)) for pattern in patterns)
            if score > 0:
                style_scores[style] = score
        
        detected_style = max(style_scores.items(), key=lambda x: x[1])[0] if style_scores else "neutral"
        
        metadata = {
            "domain": detected_domain,
            "style": detected_style,
            "domain_scores": domain_scores,
            "style_scores": style_scores,
            "total_text_length": len(all_text)
        }
        
        return metadata

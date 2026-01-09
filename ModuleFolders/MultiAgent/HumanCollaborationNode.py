"""
人机协作节点
在关键质量控制节点预留人工介入环节
"""

from typing import Dict, Any, List, Optional
from Base.Base import Base
from PyQt5.QtWidgets import QMessageBox, QTableWidget, QTableWidgetItem, QDialog, QVBoxLayout, QPushButton
from PyQt5.QtCore import Qt


class HumanCollaborationNode(Base):
    """
    人机协作节点
    在关键质量控制节点提供人工介入功能
    """
    
    def __init__(self):
        super().__init__()
        self.pending_tasks = []  # 待人工处理的任务队列
    
    def log_agent_action(self, action: str, details: str = "") -> None:
        """记录Agent执行的动作"""
        self.info(f"[HumanCollaboration] {action} {details}")

    def request_human_input(self, task_type: str, task_data: Dict[str, Any], 
                           parent_widget=None) -> Optional[Dict[str, Any]]:
        """
        请求人工输入
        
        Args:
            task_type: 任务类型（"terminology_review", "translation_review", "batch_translation_review", "error_correction"）
            task_data: 任务数据
            parent_widget: 父窗口组件
            
        Returns:
            人工输入的结果，如果取消则返回None
        """
        self.log_agent_action(f"请求人工介入: {task_type}")
        
        if task_type == "terminology_review":
            return self._handle_terminology_review(task_data, parent_widget)
        elif task_type == "translation_review":
            # 检查是否包含批量数据（review_items），如果是则转为批量处理
            if "review_items" in task_data:
                return self._handle_batch_translation_review(task_data, parent_widget)
            return self._handle_translation_review(task_data, parent_widget)
        elif task_type == "batch_translation_review":
            return self._handle_batch_translation_review(task_data, parent_widget)
        elif task_type == "error_correction":
            return self._handle_error_correction(task_data, parent_widget)
        else:
            self.warning(f"未知的任务类型: {task_type}")
            return None
    
    def _handle_terminology_review(self, task_data: Dict, parent_widget) -> Optional[Dict]:
        """
        处理术语审核任务
        支持批量术语审核（terms列表）或单个术语审核（term）
        """
        # 检查是批量审核还是单个审核
        terms = task_data.get("terms", [])
        if terms:
            # 批量审核模式
            return self._handle_batch_terminology_review(terms, parent_widget)
        else:
            # 单个术语审核模式（向后兼容）
            term = task_data.get("term", "")
            suggested_translations = task_data.get("suggested_translations", [])
            context = task_data.get("context", "")
            
            # 创建审核对话框
            msg = QMessageBox(parent_widget)
            msg.setWindowTitle("术语审核")
            msg.setText(f"请审核以下术语的翻译：\n\n术语：{term}\n上下文：{context}\n\n建议翻译：\n" + 
                       "\n".join([f"{i+1}. {t}" for i, t in enumerate(suggested_translations)]))
            
            # 添加按钮
            msg.addButton("使用建议1", QMessageBox.AcceptRole)
            if len(suggested_translations) > 1:
                msg.addButton("使用建议2", QMessageBox.AcceptRole)
            msg.addButton("手动输入", QMessageBox.ActionRole)
            msg.addButton("跳过", QMessageBox.RejectRole)
            
            result = msg.exec_()
            
            if result == QMessageBox.AcceptRole:
                # 使用第一个建议
                return {"approved_terms": [{"term": term, "translation": suggested_translations[0] if suggested_translations else ""}]}
            elif result == QMessageBox.ActionRole:
                # 手动输入
                manual_result = self._get_manual_input(term, "术语翻译")
                if manual_result:
                    return {"approved_terms": [{"term": term, "translation": manual_result.get("translation", "")}]}
            return None  # 跳过
    
    def _handle_batch_terminology_review(self, terms: List[Dict], parent_widget) -> Optional[Dict]:
        """
        批量术语审核
        terms格式: [{"term": "术语", "info": {...}}]
        """
        approved_terms = []
        
        for term_item in terms:
            term = term_item.get("term", "")
            info = term_item.get("info", {})
            suggested_translations = info.get("translation_suggestions", [])
            if not suggested_translations and info.get("translation"):
                suggested_translations = [info.get("translation")]
            context = info.get("context", "")
            
            # 为每个术语创建审核对话框
            msg = QMessageBox(parent_widget)
            msg.setWindowTitle(f"术语审核 ({terms.index(term_item) + 1}/{len(terms)})")
            msg.setText(f"请审核以下术语的翻译：\n\n术语：{term}\n上下文：{context}\n\n建议翻译：\n" + 
                       "\n".join([f"{i+1}. {t}" for i, t in enumerate(suggested_translations)]))
            
            # 添加按钮
            accept_btn = msg.addButton("接受建议", QMessageBox.AcceptRole)
            if len(suggested_translations) > 1:
                msg.addButton("使用建议2", QMessageBox.AcceptRole)
            manual_btn = msg.addButton("手动输入", QMessageBox.ActionRole)
            skip_btn = msg.addButton("跳过", QMessageBox.RejectRole)
            all_accept_btn = msg.addButton("全部接受", QMessageBox.YesRole)
            
            result = msg.exec_()
            clicked_btn = msg.clickedButton()
            
            if clicked_btn == all_accept_btn:
                # 全部接受剩余术语
                for remaining_term in terms[terms.index(term_item):]:
                    remaining_info = remaining_term.get("info", {})
                    remaining_suggestions = remaining_info.get("translation_suggestions", [])
                    if not remaining_suggestions and remaining_info.get("translation"):
                        remaining_suggestions = [remaining_info.get("translation")]
                    approved_terms.append({
                        "term": remaining_term.get("term", ""),
                        "translation": remaining_suggestions[0] if remaining_suggestions else ""
                    })
                break
            elif result == QMessageBox.AcceptRole:
                # 使用第一个建议
                approved_terms.append({
                    "term": term,
                    "translation": suggested_translations[0] if suggested_translations else ""
                })
            elif clicked_btn == manual_btn:
                # 手动输入
                manual_result = self._get_manual_input(term, "术语翻译")
                if manual_result:
                    approved_terms.append({
                        "term": term,
                        "translation": manual_result.get("translation", "")
                    })
            # 跳过的情况不添加到approved_terms
        
        if approved_terms:
            return {"approved_terms": approved_terms}
        return None
    
    def _handle_translation_review(self, task_data: Dict, parent_widget) -> Optional[Dict]:
        """
        处理翻译审核任务（单个翻译项）
        """
        source_text = task_data.get("source_text", "")
        translated_text = task_data.get("translated_text", "")
        issues = task_data.get("issues", [])
        
        msg = QMessageBox(parent_widget)
        msg.setWindowTitle("翻译审核")
        msg.setText(f"请审核以下翻译：\n\n原文：{source_text}\n\n译文：{translated_text}\n\n发现问题：\n" + 
                   "\n".join(issues))
        
        msg.addButton("接受", QMessageBox.AcceptRole)
        msg.addButton("修正", QMessageBox.ActionRole)
        msg.addButton("重新翻译", QMessageBox.RejectRole)
        
        result = msg.exec_()
        
        if result == QMessageBox.AcceptRole:
            return {"action": "accept", "translation": translated_text}
        elif result == QMessageBox.ActionRole:
            return self._get_manual_input(translated_text, "修正翻译")
        else:
            return {"action": "retranslate"}
    
    def _handle_batch_translation_review(self, task_data: Dict, parent_widget) -> Optional[Dict]:
        """
        处理批量翻译审核任务（用于回译评分未通过的行）
        
        Args:
            task_data: {
                "review_items": [{
                    "index": 行索引,
                    "source_text": 原文,
                    "translated_text": 译文,
                    "back_translation": 回译,
                    "score": 评分,
                    "context_before": 上文（可选）,
                    "context_after": 下文（可选）
                }],
                "review_threshold": 评分阈值
            }
        
        Returns:
            {"review_results": [{"index": int, "action": str, "translation": str}]}
        """
        review_items = task_data.get("review_items", [])
        
        if not review_items:
            self.warning("批量审核：没有需要审核的项目")
            return None
        
        self.info(f"批量审核：共 {len(review_items)} 行需要人工审核")
        
        try:
            # 导入并创建批量审核对话框
            from UserInterface.EditView.HumanReview.TranslationReviewDialog import TranslationReviewDialog
            
            # 使用QDialog的exec()方法同步等待用户操作
            dialog = TranslationReviewDialog(review_items, parent_widget)
            dialog_result = dialog.exec_()
            
            if dialog_result == QDialog.Accepted:
                review_results = dialog.get_review_results()
                self.info(f"用户完成审核：{len(review_results)} 行")
                return {"review_results": review_results}
            else:
                self.info("用户取消审核")
                return None
                
        except Exception as e:
            self.error(f"批量审核对话框失败: {e}", e)
            return None
    
    def _handle_error_correction(self, task_data: Dict, parent_widget) -> Optional[Dict]:
        """
        处理错误修正任务
        """
        error_type = task_data.get("error_type", "")
        error_description = task_data.get("error_description", "")
        affected_text = task_data.get("affected_text", "")
        
        msg = QMessageBox(parent_widget)
        msg.setWindowTitle("错误修正")
        msg.setText(f"发现错误：{error_type}\n\n描述：{error_description}\n\n受影响文本：{affected_text}")
        
        msg.addButton("修正", QMessageBox.AcceptRole)
        msg.addButton("忽略", QMessageBox.RejectRole)
        
        result = msg.exec_()
        
        if result == QMessageBox.AcceptRole:
            return self._get_manual_input(affected_text, "修正内容")
        else:
            return {"action": "ignore"}
    
    def _get_manual_input(self, default_value: str, prompt: str) -> Optional[Dict]:
        """
        获取手动输入（简化版）
        实际实现中可以使用更复杂的输入对话框
        """
        # 这里简化处理，实际应该使用QInputDialog或自定义对话框
        from PyQt5.QtWidgets import QInputDialog
        
        text, ok = QInputDialog.getText(None, "手动输入", f"{prompt}:", text=default_value)
        if ok and text:
            return {"translation": text, "manual": True}
        return None
    
    def create_review_table(self, review_items: List[Dict]) -> QTableWidget:
        """
        创建审核表格
        用于批量显示需要人工审核的项目
        """
        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["原文", "译文", "状态", "操作"])
        
        for row, item in enumerate(review_items):
            table.insertRow(row)
            table.setItem(row, 0, QTableWidgetItem(item.get("source", "")))
            table.setItem(row, 1, QTableWidgetItem(item.get("translated", "")))
            table.setItem(row, 2, QTableWidgetItem(item.get("status", "待审核")))
            # 操作按钮可以后续添加
        
        return table

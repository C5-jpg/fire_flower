import sys
import traceback

if __name__ == '__main__':
    try:
        # 导入并执行分析脚本
        with open('analyze_bike_data_final.py', 'r', encoding='utf-8') as f:
            script_code = f.read()
        exec(script_code)
    except Exception as e:
        print(f"脚本执行出错: {e}", file=sys.stderr)
        print("详细错误堆栈:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
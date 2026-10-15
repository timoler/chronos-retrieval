# -*- coding: utf-8 -*-
"""Isolated GPT text calls via logged-in Codex CLI; no API credential export."""
import hashlib, json, os, subprocess, time, shutil
from pathlib import Path

class CodexTextAdapter:
    def __init__(self,root,model):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.model=model
        self.node=Path(os.environ.get('PAER_NODE') or shutil.which('node') or 'node')
        self.cli=Path(os.environ.get('PAER_CODEX_CLI') or (Path(os.environ['APPDATA'])/'npm/node_modules/@openai/codex/bin/codex.js'))
        self.schema={'type':'object','properties':{'raw_response':{'type':'string'}},'required':['raw_response'],'additionalProperties':False}
    def call(self,system,user,stage,model):
        assert model==self.model
        obj={'stage':stage,'system':system,'user':user,'model':model}
        key=hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:16]
        cache=self.root/'cache'/f'{key}.json';cache.parent.mkdir(exist_ok=True)
        if cache.exists():
            result=json.loads(cache.read_text(encoding='utf-8'))
            return result['raw_response'],str(cache),True
        run=self.root/'calls'/key;run.mkdir(parents=True,exist_ok=True)
        dump=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
        dump(run/'request.json',obj);dump(run/'schema.json',self.schema)
        prompt=('You are a text-only component in a frozen forecasting experiment. '
                'Use only the SYSTEM and USER payloads below. Do not use tools, read files, browse, '
                'run commands, or access other sessions. Do not infer validation or test performance. '
                'Return the requested complete JSON object as a string in the outer raw_response field; '
                'the outer wrapper is only transport. No markdown or commentary.\n\n'
                'SYSTEM PAYLOAD:\n'+system+'\n\nUSER PAYLOAD:\n'+user)
        (run/'prompt.txt').write_text(prompt,encoding='utf-8')
        args=[str(self.node),str(self.cli),'exec','--ignore-user-config','--ephemeral',
              '--model',model,'--sandbox','read-only','--skip-git-repo-check',
              '--config','model_reasoning_effort="medium"','--json','--color','never',
              '--cd',str(run),'--output-schema',str(run/'schema.json'),
              '--output-last-message',str(run/'answer.json'),'-']
        dump(run/'invocation.json',{'model':model,'reasoning_effort':'medium','sandbox':'read-only','ephemeral':True,'tools_allowed_by_experiment':False,'docs':'https://developers.openai.com/cookbook/examples/codex/build_iterative_repair_loops_with_codex'})
        print('GPT REQUEST',stage,model,key,flush=True)
        start=time.monotonic()
        # Auth remains inside the existing Codex login; no token is copied out.
        env=os.environ.copy();env['PYTHONIOENCODING']='utf-8'
        # Honor the user's enabled Windows proxy only in this child process.
        # No system setting or login credential is changed.
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,r'Software\Microsoft\Windows\CurrentVersion\Internet Settings') as reg:
            enabled=winreg.QueryValueEx(reg,'ProxyEnable')[0]
            proxy=winreg.QueryValueEx(reg,'ProxyServer')[0] if enabled else ''
        if proxy and '=' not in proxy and not env.get('HTTPS_PROXY'):
            env['HTTPS_PROXY']='http://'+proxy
            env['HTTP_PROXY']='http://'+proxy
        with (run/'stdout.jsonl').open('w',encoding='utf-8') as stdout,(run/'stderr.log').open('w',encoding='utf-8') as stderr:
            proc=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=stdout,stderr=stderr,text=True,
                                  encoding='utf-8',env=env,creationflags=subprocess.CREATE_NO_WINDOW)
            try: proc.communicate(prompt,timeout=900)
            except subprocess.TimeoutExpired:
                proc.kill();proc.wait();raise RuntimeError('GPT request timed out; do not regenerate by score')
        if proc.returncode!=0:
            raise RuntimeError(f'GPT CLI failed (exit {proc.returncode}); inspect saved non-secret CLI status')
        events=[]
        for line in (run/'stdout.jsonl').read_text(encoding='utf-8').splitlines():
            try: events.append(json.loads(line))
            except json.JSONDecodeError: continue
        prohibited={'command_execution','mcp_tool_call','web_search','file_change','collab_tool_call'}
        assert not any(e.get('item',{}).get('type') in prohibited for e in events),'GPT used a tool; invalidate this call'
        answer=json.loads((run/'answer.json').read_text(encoding='utf-8'))
        raw=answer['raw_response']
        assert raw.strip()
        obj['raw_response']=raw;obj['transport']='codex exec ephemeral JSON wrapper'
        obj['elapsed_seconds']=time.monotonic()-start
        obj['usage']=[e.get('usage') for e in events if e.get('type')=='turn.completed']
        obj['tools_used']=False
        dump(cache,obj)
        print('GPT RESPONSE',stage,'elapsed',round(obj['elapsed_seconds'],1),'usage',obj['usage'],flush=True)
        return raw,str(cache),False

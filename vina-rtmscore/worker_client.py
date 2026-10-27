"""Persistent JSON-lines scorer process and owned-process cleanup."""
import json
from pathlib import Path
import selectors
import subprocess

class RTMClient:
    def __init__(self,python,script,rtm_root,directory,timeout=300.):
        self.directory=Path(directory);self.timeout=timeout;self.calls=0
        self.log=(self.directory/'rtm_worker.log').open('w')
        self.process=subprocess.Popen([str(python),'-u',str(script),'--rtm-root',str(rtm_root)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1)
        self.selector=selectors.DefaultSelector()
        self.selector.register(self.process.stdout,selectors.EVENT_READ)

    def request(self,payload):
        self.process.stdin.write(json.dumps(payload)+'\n');self.process.stdin.flush()
        if not self.selector.select(self.timeout):
            raise TimeoutError('RTMScore worker timed out; inspect rtm_worker.log')
        line=self.process.stdout.readline()
        if not line: raise RuntimeError('RTMScore worker exited; inspect rtm_worker.log')
        response=json.loads(line)
        if 'error' in response: raise RuntimeError('RTMScore worker: '+response['error'])
        return response


    def close(self):
        try:
            try:self.process.stdin.close()
            except (BrokenPipeError,OSError):pass
            try:self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        finally:
            self.process.stdout.close()
            self.selector.close();self.log.close()

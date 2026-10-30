import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'app'))
from chat_queue import ChatQueue

class QueueTests(unittest.TestCase):
    def test_limit_fifo_and_positions(self):
        q=ChatQueue()
        a,b,c,d=[q.join() for _ in range(4)]
        self.assertTrue(q.claim(a)); self.assertTrue(q.claim(b))
        self.assertFalse(q.claim(c)); self.assertEqual(q.position(c),1)
        self.assertEqual(q.position(d),2)
        q.release(a)
        self.assertFalse(q.claim(d)); self.assertTrue(q.claim(c))
        self.assertEqual(len(q.active),2)
    def test_disconnect_removes_waiter_and_release_is_idempotent(self):
        q=ChatQueue();a,b,c=[q.join() for _ in range(3)]
        q.claim(a);q.claim(b);q.release(c);q.release(c)
        self.assertEqual(len(q.waiting),0)
        q.release(a);d=q.join();self.assertTrue(q.claim(d))

if __name__=='__main__': unittest.main()

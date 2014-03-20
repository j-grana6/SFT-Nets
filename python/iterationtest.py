def itertest(a):
    while True:
        try:
            a.next()
        except StopIteration:
            print 'done'
            break
    return 'worked'

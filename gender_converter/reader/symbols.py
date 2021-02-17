id2sp = None
id2ph = None


def get_experiment_phn_info():
    """
    Retrieve experiment specific phoneme information

    Returns:  phone_list, ph2id, id2ph

    """
    phone_list = ['##', 'aa', 'ae', 'ao', 'aw', 'ax', 'ay', 'bb', 'br',
                  'ch', 'dd', 'dh', 'eh', 'er', 'ey', 'ff', 'gg', 'hh', 'ih',
                  'iy', 'jh', 'kk', 'll', 'mm', 'ng', 'nn', 'ow', 'oy', 'pp',
                  'rr', 'sh', 'sp', 'ss', 'th', 'tt', 'uh', 'uw', 'vv', 'ww',
                  'yy', 'zh', 'zz']
    ph2id = {ph: i for i, ph in enumerate(phone_list)}
    id2ph = {i: ph for i, ph in enumerate(phone_list)}

    return phone_list, ph2id, id2ph


def get_experiment_speaker_info(db_root):
    """
    return a tuple containing all the experiment specific speaker related information

    Args:
        db_root

    Returns:
        seen_speakers, sp2id, id2sp

    """
    seen_speakers = ['VCTK-speaker-p225-female',
 'VCTK-speaker-p226-male',
 'VCTK-speaker-p227-male',
 'VCTK-speaker-p228-female',
 'VCTK-speaker-p229-female',
 'VCTK-speaker-p230-female',
 'VCTK-speaker-p231-female',
 'VCTK-speaker-p232-male',
 'VCTK-speaker-p233-female',
 'VCTK-speaker-p234-female',
 'VCTK-speaker-p236-female',
 'VCTK-speaker-p237-male',
 'VCTK-speaker-p238-female',
 'VCTK-speaker-p239-female',
 'VCTK-speaker-p240-female',
 'VCTK-speaker-p241-male',
 'VCTK-speaker-p243-male',
 'VCTK-speaker-p244-female',
 'VCTK-speaker-p245-male',
 'VCTK-speaker-p246-male',
 'VCTK-speaker-p247-male',
 'VCTK-speaker-p248-female',
 'VCTK-speaker-p249-female',
 'VCTK-speaker-p250-female',
 'VCTK-speaker-p251-male',
 'VCTK-speaker-p252-male',
 'VCTK-speaker-p253-female',
 'VCTK-speaker-p254-male',
 'VCTK-speaker-p255-male',
 'VCTK-speaker-p256-male',
 'VCTK-speaker-p257-female',
 'VCTK-speaker-p258-male',
 'VCTK-speaker-p259-male',
 'VCTK-speaker-p260-male',
 'VCTK-speaker-p261-female',
 'VCTK-speaker-p262-female',
 'VCTK-speaker-p263-male',
 'VCTK-speaker-p264-female',
 'VCTK-speaker-p265-female',
 'VCTK-speaker-p266-female',
 'VCTK-speaker-p267-female',
 'VCTK-speaker-p268-female',
 'VCTK-speaker-p269-female',
 'VCTK-speaker-p270-male',
 'VCTK-speaker-p271-male',
 'VCTK-speaker-p272-male',
 'VCTK-speaker-p273-male',
 'VCTK-speaker-p274-male',
 'VCTK-speaker-p275-male',
 'VCTK-speaker-p276-female',
 'VCTK-speaker-p277-female',
 'VCTK-speaker-p278-male',
 'VCTK-speaker-p279-male',
 'VCTK-speaker-p280-female',
 'VCTK-speaker-p281-male',
 'VCTK-speaker-p282-female',
 'VCTK-speaker-p283-female',
 'VCTK-speaker-p284-male',
 'VCTK-speaker-p285-male',
 'VCTK-speaker-p286-male',
 'VCTK-speaker-p287-male',
 'VCTK-speaker-p288-female',
 'VCTK-speaker-p292-male',
 'VCTK-speaker-p293-female',
 'VCTK-speaker-p294-female',
 'VCTK-speaker-p295-female',
 'VCTK-speaker-p297-female',
 'VCTK-speaker-p298-male',
 'VCTK-speaker-p299-female',
 'VCTK-speaker-p300-female',
 'VCTK-speaker-p301-female',
 'VCTK-speaker-p302-male',
 'VCTK-speaker-p303-female',
 'VCTK-speaker-p304-male',
 'VCTK-speaker-p305-female',
 'VCTK-speaker-p306-female',
 'VCTK-speaker-p307-female',
 'VCTK-speaker-p308-female',
 'VCTK-speaker-p310-female',
 'VCTK-speaker-p311-male',
 'VCTK-speaker-p312-female',
 'VCTK-speaker-p313-female',
 'VCTK-speaker-p314-female',
 'VCTK-speaker-p316-male',
 'VCTK-speaker-p317-female',
 'VCTK-speaker-p318-female',
 'VCTK-speaker-p323-female',
 'VCTK-speaker-p326-male',
 'VCTK-speaker-p329-female',
 'VCTK-speaker-p330-female',
 'VCTK-speaker-p333-female',
 'VCTK-speaker-p334-male',
 'VCTK-speaker-p335-female',
 'VCTK-speaker-p336-female',
 'VCTK-speaker-p339-female',
 'VCTK-speaker-p340-female',
 'VCTK-speaker-p341-female',
 'VCTK-speaker-p343-female',
 'VCTK-speaker-p345-male',
 'VCTK-speaker-p347-male',
 'VCTK-speaker-p351-female',
 'VCTK-speaker-p360-male',
 'VCTK-speaker-p361-female',
 'VCTK-speaker-p362-female',
 'VCTK-speaker-p363-male',
 'VCTK-speaker-p364-male',
 'VCTK-speaker-p374-male',
 'VCTK-speaker-p376-male']

    # speaker index list for training and validation
    n_speaker = len(seen_speakers)

    # take all speakers in train and validation!!!
    train_speakers = seen_speakers
    valid_speakers = seen_speakers
    print('number of VCTK speakers = %d' % n_speaker)

    sp2id = {sp: i for i, sp in enumerate(seen_speakers)}
    id2sp = {i: sp for i, sp in enumerate(seen_speakers)}

    return seen_speakers, sp2id, id2sp
